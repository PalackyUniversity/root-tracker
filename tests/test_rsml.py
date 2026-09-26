"""RSML preserves foreign data without granting it pipeline authority."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from root_tracker.io.rsml import read_rsml, save_rsml

EXTENDED = b'''<?xml version="1.0"?>
<rsml xmlns="urn:rsml" xmlns:v="urn:vendor"><metadata><unit>cm</unit>
<resolution>100</resolution><image><name>missing.png</name></image></metadata>
<!-- retain this comment exactly --><scene><plant id="p">
<root id="main"><geometry><polyline><point x="1.5" y="2" z="3"/>
<point x="4.5" y="6" z="3"/></polyline></geometry>
<functions><function name="diameter"><sample value="2"/></function></functions>
<v:unknown custom="yes">keep me</v:unknown>
<root id="lateral"><geometry><polyline><point x="4.5" y="6"/>
<point x="7" y="8"/></polyline></geometry></root></root>
</plant></scene></rsml>'''


class RSMLTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'source.rsml'
        self.output = Path(self.tmp.name) / 'copy.rsml'

    def read(self, data=EXTENDED):
        self.source.write_bytes(data)
        return read_rsml(self.source)

    def test_lossless_snapshot_with_nested_geometry(self):
        doc = self.read()
        self.source.write_bytes(b'changed')
        save_rsml(doc, self.output)
        self.assertEqual(self.output.read_bytes(), EXTENDED)
        self.assertEqual(doc.roots[1].parent_id, 'main')
        self.assertEqual(doc.roots[0].points, ((1.5, 2., 3.), (4.5, 6., 3.)))
        self.assertEqual(doc.roots[0].length, 5.)
        self.assertEqual(doc.image_name, 'missing.png')
        self.assertEqual(doc.unit, 'cm')
        self.assertEqual(doc.resolution, 100.)

    def test_utf16_is_preserved(self):
        raw = EXTENDED.decode().replace('<?xml version="1.0"?>',
              '<?xml version="1.0" encoding="UTF-16"?>').encode('utf-16')
        save_rsml(self.read(raw), self.output)
        self.assertEqual(self.output.read_bytes(), raw)

    def test_rejects_dtd_in_both_encodings(self):
        text = '<!DOCTYPE rsml [<!ENTITY x SYSTEM "file:///etc/passwd">]><rsml><scene/></rsml>'
        for encoding in ('utf-8', 'utf-16'):
            with self.subTest(encoding=encoding), self.assertRaises(ValueError):
                self.read(text.encode(encoding))

    def test_invalid_xml_and_coordinates(self):
        for raw in (b'<rsml>', b'<wrong><scene/></wrong>', b'<rsml/>',
                    EXTENDED.replace(b'x="1.5"', b'x="nan"'),
                    EXTENDED.replace(b'x="1.5"', b'x="oops"')):
            with self.subTest(raw=raw[:40]), self.assertRaises(ValueError):
                self.read(raw)

    def test_vendor_geometry_not_mistaken_for_core(self):
        raw = b'<rsml xmlns:v="urn:vendor"><scene><plant id="a"><v:root><v:geometry/></v:root><root id="r"><geometry><v:polyline/></geometry></root></plant></scene></rsml>'
        doc = self.read(raw)
        self.assertEqual(len(doc.roots), 1)
        self.assertEqual(doc.roots[0].points, ())
        save_rsml(doc, self.output)
        self.assertEqual(self.output.read_bytes(), raw)

    def test_failed_save_does_not_destroy_destination(self):
        doc = self.read()
        self.output.write_bytes(b'existing')
        with patch('root_tracker.io.rsml.os.replace', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                save_rsml(doc, self.output)
        self.assertEqual(self.output.read_bytes(), b'existing')
        self.assertEqual(len(list(self.output.parent.iterdir())), 2)

    def test_limits_and_empty_scene(self):
        with patch('root_tracker.io.rsml.MAX_BYTES', 16), self.assertRaises(ValueError):
            self.read()
        self.assertEqual(self.read(b'<rsml><scene/></rsml>').roots, ())
        with self.assertRaises(ValueError):
            self.read(b'<rsml><scene>' + b'<x>' * 300 + b'</x>' * 300 + b'</scene></rsml>')


if __name__ == '__main__':
    unittest.main()
