import unittest

import torch

from flashinfer.prefill import (
    _check_custom_mask_length,
    _compute_mask_indptr,
    _compute_page_mask_indptr,
    _compute_packed_mask_length,
)


class TestCustomMaskContract(unittest.TestCase):
    def testPagedCustomMaskLengthMatchesDerivedSpan(self):
        qo_indptr = torch.tensor([0, 2, 5], dtype=torch.int32)
        paged_kv_indptr = torch.tensor([0, 1, 3], dtype=torch.int32)
        paged_kv_last_page_len = torch.tensor([4, 2], dtype=torch.int32)

        mask_indptr = _compute_page_mask_indptr(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_last_page_len,
            page_size=4,
        )

        self.assertEqual([0, 8, 26], mask_indptr.tolist())
        _check_custom_mask_length(torch.ones(26, dtype=torch.bool), None, mask_indptr)

        with self.assertRaisesRegex(ValueError, "custom_mask length"):
            _check_custom_mask_length(
                torch.ones(25, dtype=torch.bool), None, mask_indptr
            )

    def testPagedPackedCustomMaskLengthMatchesSegmentPackedSpan(self):
        mask_indptr = torch.tensor([0, 8, 26], dtype=torch.int32)

        self.assertEqual(4, _compute_packed_mask_length(mask_indptr))
        _check_custom_mask_length(None, torch.ones(4, dtype=torch.uint8), mask_indptr)

        with self.assertRaisesRegex(ValueError, "packed_custom_mask length"):
            _check_custom_mask_length(
                None, torch.ones(3, dtype=torch.uint8), mask_indptr
            )

    def testRaggedCustomMaskLengthMatchesDerivedSpan(self):
        qo_indptr = torch.tensor([0, 2, 5], dtype=torch.int32)
        kv_indptr = torch.tensor([0, 4, 10], dtype=torch.int32)

        mask_indptr = _compute_mask_indptr(qo_indptr, kv_indptr)

        self.assertEqual([0, 8, 26], mask_indptr.tolist())
        _check_custom_mask_length(torch.ones(26, dtype=torch.bool), None, mask_indptr)

        with self.assertRaisesRegex(ValueError, "custom_mask length"):
            _check_custom_mask_length(
                torch.ones(27, dtype=torch.bool), None, mask_indptr
            )


if __name__ == "__main__":
    unittest.main()
