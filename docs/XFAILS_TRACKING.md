# FlashInfer Xfails Tracking System

This directory contains tools and reports for tracking test skips and expected failures (xfails) in the FlashInfer test suite.

## Files

- **`XFAILS_REPORT.md`** - Comprehensive human-readable report of all test issues
- **`scripts/generate_xfails_report.py`** - Automated report generation script
- **`.github/workflows/track_xfails.yml`** - CI workflow for automated tracking

## Usage

### Generate Report Manually

```bash
# Generate markdown report
python scripts/generate_xfails_report.py

# Generate with custom output file
python scripts/generate_xfails_report.py --output my_report.md

# Generate JSON format for programmatic use
python scripts/generate_xfails_report.py --format json --output xfails.json

# Generate CSV format for spreadsheet analysis
python scripts/generate_xfails_report.py --format csv --output xfails.csv

# Verbose output
python scripts/generate_xfails_report.py --verbose
```

### Automated Tracking

The GitHub Actions workflow automatically:
- Runs weekly to track changes over time
- Triggers on changes to test files
- Commits updated reports to the repository
- Provides downloadable artifacts

## Report Categories

The system categorizes xfails/skips into:

1. **Hardware Requirements** - GPU compute capability and architecture issues
2. **Feature Unsupported** - Missing functionality or incomplete implementations
3. **Parameter Validation** - Invalid parameter combinations or validation issues
4. **Environment Issues** - Memory, device, or platform limitations
5. **Backend Limitations** - Backend-specific restrictions (CUDNN, Cutlass, TensorRT-LLM)
6. **Other** - Miscellaneous issues

## Developer Workflow

### For Fixing Issues

1. Review the current `XFAILS_REPORT.md`
2. Pick issues from high-priority categories
3. Fix the underlying problem
4. Remove the corresponding `pytest.skip()` or `@pytest.mark.xfail`
5. Run tests to ensure they pass
6. Regenerate report to track progress

### For Adding New Tests

1. If adding a test that should be skipped on certain conditions:
   - Use descriptive reasons: `pytest.skip("Specific reason for skip")`
   - Follow existing patterns for similar issues
   - Consider if the skip is temporary or permanent

### For Code Reviews

1. Check if PRs introduce new xfails/skips
2. Ensure new skips have proper justification
3. Consider if skips should be conditional rather than absolute

## Monitoring Progress

Key metrics to track:
- **Total xfail count** - Overall test health
- **Category distributions** - Which types of issues are most common
- **File coverage** - Which test files have the most issues
- **Trend over time** - Are we reducing technical debt?

## Integration with CI/CD

The tracking system can be enhanced with:
- Automatic issue creation for xfail regressions
- PR comments showing xfail impact
- Dashboard visualization of trends
- Integration with test result reporting

## Contributing

When modifying the tracking system:
1. Test changes with `python scripts/generate_xfails_report.py --verbose`
2. Ensure backward compatibility for existing reports
3. Update this README if adding new features
4. Consider performance impact for large test suites

## Future Enhancements

Planned improvements:
- [ ] Historical trend analysis
- [ ] Integration with test result databases
- [ ] Automatic priority scoring for issues
- [ ] Cross-reference with GitHub issues
- [ ] Performance benchmarking for fix validation
- [ ] Integration with hardware CI matrix