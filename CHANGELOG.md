# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),

# [1.1.0] 2024-07-15

### Added

- Add GLONASS FDMA, NavIC, BDS D1/D2, CNAV-2/3 message decoder
- Add GLONASS frequency channel number to apc2com() (@AndreHauschild)
- Add a test workflow (@AndreHauschild)
- Add APC reference corrections for IGS and RTCM3 SSR corrections (@AndreHauschild)
- Decoder for Javad receiver (on cssrlib-data)

### Changed

- Use different APC reference signals for SIS and IDD of Galileo HAS (@AndreHauschild)
- Change MADOCA APC reference (@AndreHauschild)

### Fixed

- Fixed CNAV decoder

# [1.0.0] 2024-01-01

### Added
New integrated class structure (PPPOS) for PPP/PPP-RTK/RTK processing

- Support for RTCM3 (Galileo HAS IDD)
- Decoder for Septentrio receiver (on cssrlib-data)
- Parser for RTCM3, L5 SBAS
- Experimental support for PPP via SouthPAN (PVS)
- New solid Earth tides model using PySolid (2010 IERS Conventions) 

### Changed

- Improved documentation
- Sign of SSR satellite signal code/phase bias align with RTCM 3 convention
- PPPIGS was integrated into PPPOS and removed

### Fixed

- Link for cssrlib-data

### Deprecated
### Removed
Function based PPP/PPP-RTK/RTK processing
### Security

# [0.8.0] 2023-09-09

### Added

- New signal structure
- Support for open PPP services: Galileo HAS (SIS), BDS PPP, QZSS MADOCA-PPP
- Support for IGS (SP3+BIAS)
- Parser for SP3, ANTEX, BIAS files
- Support for PPP-AR
- Jupyter notebook with examples

### Changed

Improved documentation

Added link for Google Colab

### Fixed
### Deprecated
### Removed
### Security

# [0.3.0] 2022-03-01

### Added
Initial version for PPP-RTK (QZSS CLAS) and RTK

### Changed
### Fixed
### Deprecated
### Removed
### Security
