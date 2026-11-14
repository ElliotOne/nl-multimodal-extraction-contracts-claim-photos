# Source photo attribution

The sample claim photos are derived from two photographs published on Wikimedia Commons.

| File in this folder | Source | Author | Licence |
| --- | --- | --- | --- |
| `abandoned-suzuki-osidge-london-1.jpg` | [File:Abandoned Suzuki, Osidge, London.jpg](https://commons.wikimedia.org/wiki/File:Abandoned_Suzuki,_Osidge,_London.jpg) | [Philafrenzy](https://commons.wikimedia.org/wiki/User:Philafrenzy) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `abandoned-suzuki-osidge-london-2.jpg` | [File:Abandoned Suzuki, Osidge, London (2).jpg](https://commons.wikimedia.org/wiki/File:Abandoned_Suzuki,_Osidge,_London_(2).jpg) | [Philafrenzy](https://commons.wikimedia.org/wiki/User:Philafrenzy) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |

Both originals were taken on 2017-07-16 in Osidge, London. They are used here as a
realistic example of a crash-damaged vehicle for an educational software project.

## Modifications

The files in this folder are downscaled to 2,400 pixels on the long side with the
original EXIF removed. `scripts/make_sample_photos.py` derives the six-photo sample
upload set under `data/claims/CLM-2026-004471/photos/` from them: crops, a blurred
copy, a re-encoded copy, a rotated copy, a thumbnail, and rewritten EXIF timestamps,
GPS coordinates, and orientation tags that fit the fictional claim. The fictional claim
record (vehicle year, incident date, claimant statement) is not a statement about the
real vehicle or its owner.

All derivatives are distributed under the same CC BY-SA 4.0 licence. The MIT licence
in the repository root covers the source code only.
