import unittest
from dgx_spark_sm121_audit import audit_dgx_spark_sm121_support

class TestDgxSparkSm121Audit(unittest.TestCase):
    def test_sm121_supported(self):
        cuda_version = "12.9"
        audit_dgx_spark_sm121_support()
        self.assertTrue(audit_dgx_spark_sm121_support())

    def test_sm121_not_supported(self):
        cuda_version = "12.8"
        audit_dgx_spark_sm121_support()
        self.assertFalse(audit_dgx_spark_sm121_support())

if __name__ == "__main__":
    unittest.main()