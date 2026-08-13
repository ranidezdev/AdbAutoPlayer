# Changelog

## [12.9.29] - 2026-08-13

### Bug Fixes

- **Device Streaming**: Fixed emulator detection on HyperOS (POCO X5 Pro) where
  `ro.kernel.qemu` is present but set to `0`, incorrectly identifying physical
  devices as emulators.
- **CLI**: Fixed logging format in CLI exception handling to properly pass the
  exception object for traceback logging.
- **AFK Journey**: Fixed exception logging format in formation copying.

### Features

- **Game Engine**: Added `diagnostic_recheck` parameter to
  `wait_for_template` / `wait_for_any_template` to perform an extra check after
  timeout and suggest a proper timeout value, helping debug slow navigation
  steps.
- **AFK Journey**: Enabled diagnostic recheck for the formation selection
  screen and added debug screenshot capture on failure.
