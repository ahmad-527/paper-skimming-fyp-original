"""Optional real-checkpoint regressions; weights remain outside the repository."""
import json
import os
from pathlib import Path
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from server import Classifier, LABELS, position_features
from exports import readout_exports


class ExportTests(unittest.TestCase):
    def test_recorded_export_preserves_every_sentence_and_score(self):
        sample = json.loads((Path(__file__).parent / 'static/sample.json').read_text())
        original = json.dumps(sample, sort_keys=True)
        files = readout_exports(sample)
        exported = json.loads(files['json'])
        self.assertEqual(exported['sentences'], sample['sentences'])
        self.assertEqual(exported['source'], sample['source'])
        self.assertEqual(exported['mode'], 'recorded')
        self.assertFalse(exported['scores_are_calibrated'])
        self.assertEqual(json.dumps(sample, sort_keys=True), original)
        markdown = files['md'].decode('utf-8')
        self.assertIn('Preserved notebook example', markdown)
        for row in sample['sentences']:
            self.assertIn(row['text'], markdown)
            self.assertIn(f"## {row['index'] + 1}. {row['label']}", markdown)

    def test_live_export_preserves_unicode_and_provenance(self):
        text = 'β-catenin changed by 3.5 µg/mL. <script>original wording</script>'
        readout = {'mode': 'live', 'model': 'Historical checkpoint', 'text': text,
                   'sentences': [{'index': 0, 'text': text, 'label': 'RESULTS',
                                  'scores': {label: float(label == 'RESULTS') for label in LABELS}}]}
        files = readout_exports(readout)
        self.assertEqual(json.loads(files['json'])['text'], text)
        self.assertIn(text, files['md'].decode('utf-8'))
        self.assertIn('Source: Historical checkpoint', files['md'].decode('utf-8'))
        self.assertIn('randomized controlled trial', files['md'].decode('utf-8'))

    def test_export_rejects_nonfinite_probabilities(self):
        with self.assertRaises(ValueError):
            readout_exports({'mode': 'live', 'model': 'Historical', 'sentences': [], 'value': float('nan')})


class HistoricalInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get('PAPER_SKIMMING_MODEL')
        if not path:
            raise unittest.SkipTest('Set PAPER_SKIMMING_MODEL to the external historical checkpoint.')
        cls.classifier = Classifier(path)
        cls.classifier.load()
        if cls.classifier.state != 'ready':
            raise RuntimeError('Historical checkpoint did not load.')

    def check_readout(self, text, expected_count):
        result = self.classifier.predict(text)
        self.assertEqual(result['mode'], 'live')
        self.assertEqual(len(result['sentences']), expected_count)
        self.assertEqual(result['text'], text.strip())
        for row in result['sentences']:
            self.assertEqual(set(row['scores']), set(LABELS))
            self.assertTrue(all(0 <= score <= 1 for score in row['scores'].values()))
            self.assertAlmostEqual(sum(row['scores'].values()), 1, places=4)
        return result

    def test_graph_matches_original_eager_inference(self):
        sample = json.loads((Path(__file__).parent / 'static/sample.json').read_text())
        sentences = [row['text'] for row in sample['sentences']]
        tf = self.classifier.tf
        lines, totals = position_features(len(sentences))
        inputs = [tf.constant(lines, tf.int32), tf.constant(totals, tf.int32),
                  tf.constant(sentences), tf.constant([[' '.join(sentence)] for sentence in sentences])]
        eager = self.classifier.model(inputs, training=False).numpy()
        graph = self.classifier.inference(inputs).numpy()
        self.assertLessEqual(float(abs(eager - graph).max()), 1e-5)
        self.assertEqual(eager.argmax(axis=1).tolist(), graph.argmax(axis=1).tolist())

    def test_variable_lengths_and_pasted_formatting(self):
        for count in (1, 5, 16, 21, 80):
            with self.subTest(sentences=count):
                text = ' '.join(f'Measurement {index + 1} was recorded.' for index in range(count))
                result = self.check_readout(text, count)
                self.assertEqual(result['position_note'], count > 15)
        self.check_readout('BACKGROUND: Changes in β-catenin were measured.\n\nRESULTS: The mean was 3.5 µg/mL (95% CI 2.1–4.9).', 2)
        self.check_readout('A' * 20_000, 1)

    def test_input_limits_and_recovery(self):
        for value in ('', ' ', None, 'x' * 20_001, 'A measurement was recorded. ' * 81, '\ud800'):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(ValueError):
                    self.classifier.predict(value)
        self.check_readout('The trial measured walking distance. Participants were assigned at random.', 2)

    def test_busy_rejection_and_inference_failure_recovery(self):
        self.classifier.lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'another abstract'):
                self.classifier.predict('The trial recorded blood pressure.')
        finally:
            self.classifier.lock.release()
        original = self.classifier._predict
        try:
            def failure(sentences):
                raise RuntimeError('Injected model failure')
            self.classifier._predict = failure
            with self.assertRaisesRegex(RuntimeError, 'Injected model failure'):
                self.classifier.predict('The trial recorded blood pressure.')
            self.assertFalse(self.classifier.lock.locked())
            import numpy as np
            self.classifier._predict = lambda sentences: np.zeros((0, 5))
            with self.assertRaisesRegex(RuntimeError, 'invalid prediction'):
                self.classifier.predict('The trial recorded blood pressure.')
        finally:
            self.classifier._predict = original
        self.check_readout('The trial recorded blood pressure.', 1)

    def test_repeated_new_caller_threads(self):
        for _ in range(3):
            with ThreadPoolExecutor(max_workers=1) as caller:
                caller.submit(self.check_readout, 'A trial was conducted. The outcome was measured.', 2).result(timeout=30)

    def test_survives_concurrent_streamlit_keras_cleanup(self):
        # Streamlit 1.50 clears Keras state after each script run. Previously this
        # interrupted another visitor's eager RNN and raised NoneType.pop.
        text = ' '.join(f'The trial recorded measurement {index + 1} in the intervention group.' for index in range(80))
        for _ in range(3):
            started = threading.Event()

            def visitor():
                started.set()
                return self.check_readout(text, 80)

            with ThreadPoolExecutor(max_workers=1) as caller:
                pending = caller.submit(visitor)
                self.assertTrue(started.wait(5))
                cleanups = 0
                deadline = time.monotonic() + 30
                while not pending.done() and time.monotonic() < deadline:
                    self.classifier.tf.keras.backend.clear_session(free_memory=False)
                    cleanups += 1
                    time.sleep(0.01)
                pending.result(timeout=5)
                self.assertGreater(cleanups, 0)


if __name__ == '__main__':
    unittest.main()
