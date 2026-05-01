# FFmpeg Corresponding Source (Bundled Binaries)

This repository includes prebuilt FFmpeg executables under `vendor/ffmpeg/bin/`
for use by packaged Windows GUI builds.

## What is bundled

The bundled binaries come from `www.gyan.dev` and are identified in
`vendor/ffmpeg/README.txt` as:

- Version/build: `8.1-essentials_build-www.gyan.dev`
- License: GPL v3

## License and source availability

These FFmpeg binaries are licensed under the GNU General Public License, version
3 (GPLv3). When distributing these binaries, GPLv3 requires providing the
**Corresponding Source** for the exact binaries being distributed.

“Corresponding Source” is more than a link to upstream FFmpeg: it is the source
needed to generate the exact distributed binaries, including relevant build
configuration/scripts and any source required by the build (for example, where
the build incorporates additional libraries).

## Where to obtain the Corresponding Source for releases of this project

For each published Windows GUI release of Audio Analyser that bundles FFmpeg:

- The release assets include a separate source archive named similar to
  `ffmpeg-corresponding-source-<version>.zip`, or an equivalent clearly-labeled
  archive, containing the Corresponding Source for the FFmpeg binaries shipped
  with that release.

In this repository, a convenience copy of the FFmpeg upstream source for the
referenced commit may exist as:

- `vendor/ffmpeg/FFmpeg-9047fa1b084f76b1b4d065af2d743df1b40dfb56.zip`

Note: this file is **not** bundled into the packaged Windows GUI distribution.
It is intended to be used as (part of) the release-side Corresponding Source
delivery.

If you obtained the binary distribution from another source and do not have the
Corresponding Source archive, open an issue on the project repository or contact
the distributor of that binary to obtain the Corresponding Source.

