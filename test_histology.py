import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import histology_pipeline as core

ROOT = Path(__file__).resolve().parent

class RestartTests(unittest.TestCase):
    def test_red_channel_labels_survive(self):
        labels = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
        rgba = np.zeros((2, 3, 4), dtype=np.uint8)
        rgba[:, :, 0] = labels
        rgba[:, :, 3] = 255
        np.testing.assert_array_equal(core.mask_labels(rgba, 'radboud'), labels)
        with self.assertRaises(ValueError):
            core.mask_labels(rgba, 'karolinska')
        rgba[0, 0, 3] = 0
        with self.assertRaises(ValueError):
            core.mask_labels(rgba, 'radboud')

    def test_provider_specific_cancer_labels(self):
        a = np.array([[0, 1, 2, 3, 4, 5]])
        np.testing.assert_array_equal(core.target_pixels(a, 'radboud', 'cancer'), [[0, 0, 0, 1, 1, 1]])
        np.testing.assert_array_equal(core.target_pixels(a[:, :3], 'karolinska', 'cancer'), [[0, 0, 1]])

    def test_anisotropic_physical_size_and_edges(self):
        self.assertEqual(core.physical_patch_size(500, 0.5, 0.25), (1000, 2000))
        self.assertEqual(core.axis_origins(25, 10, 6), [0, 6, 12, 15])
        for bad in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                core.physical_patch_size(500, bad, .5)

    def test_missing_pair_stops_before_openslide(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            root = Path(d)
            (root/'train_images').mkdir()
            (root/'train_images/a.tiff').touch()
            with patch.object(core.openslide, 'OpenSlide') as reader:
                with self.assertRaisesRegex(FileNotFoundError, 'a_mask.tiff'):
                    core.open_pair(root, 'a')
                reader.assert_not_called()

    def test_nested_mser_keeps_larger_outline(self):
        yy,xx=np.mgrid[4:10,4:10]
        outer=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.int32)
        yy,xx=np.mgrid[6:9,6:9]
        inner=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.int32)
        rgb=np.zeros((20,20,3),np.uint8)
        annotation=np.full((20,20),4,np.uint8)
        signal=np.full((20,20),.8)
        signal[6:9,6:9]=1.0
        with patch.object(core,'hematoxylin_signal',return_value=signal), \
             patch.object(core,'threshold_otsu',return_value=.5), \
             patch.object(core.cv2,'MSER_create') as create:
            create.return_value.detectRegions.return_value=([inner,outer],None)
            instances,parameters=core.mser_candidates(rgb,annotation,'radboud',(1,1))
        self.assertEqual(parameters['objects'],1)
        self.assertEqual(np.count_nonzero(instances),36)

    def test_svg_units_holes_and_individual_frame(self):
        labels = np.zeros((20, 30), np.int32)
        labels[2:12, 3:15] = 1
        labels[5:8, 6:9] = 0
        labels[15:18, 20:25] = 2
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            dest = Path(d)/'svg'
            self.assertEqual(core.export_svgs(labels, (.5, .25), dest, {'source':'synthetic test'}), 2)
            ns = {'s':'http://www.w3.org/2000/svg'}
            svg = core.ET.parse(dest/'combined.svg').getroot()
            self.assertEqual(svg.attrib['viewBox'], '0 0 15.000000 5.000000')
            self.assertEqual(svg.attrib['width'], '0.015000000mm')
            paths = svg.findall('s:path', ns)
            self.assertEqual(len(paths), 2)
            self.assertEqual(paths[0].attrib['d'].count('M '), 2)
            self.assertEqual(paths[0].attrib['d'].count(' Z'), 2)
            self.assertEqual(paths[0].attrib['fill-rule'], 'evenodd')
            pairs = np.array([(float(x),float(y)) for x,y in re.findall(r'([\d.]+),([\d.]+)', paths[0].attrib['d'])])
            np.testing.assert_allclose(pairs.min(axis=0), [1.5, .5])
            np.testing.assert_allclose(pairs.max(axis=0), [7.5, 3.0])
            individual = core.ET.parse(dest/'object_0001.svg').getroot()
            self.assertEqual(individual.attrib, svg.attrib)
            self.assertEqual(individual.find('s:path',ns).attrib['d'], paths[0].attrib['d'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
