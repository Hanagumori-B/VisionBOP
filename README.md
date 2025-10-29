<h1 align="center">VISION BOP</h1>
<h2 align="center">---Export Vision6D to BOP Datasets---</h2>

## INTRODUCTION

**This project is based on the [Vision6D 0.5.4 version](https://github.com/InteractiveGL/vision6D/tree/0.5.4) and has added features such as *loading, saving and switching workspace* and *exporting datasets in bop format*. The README file of the original project has been preserved as [ORIGINAL_README.md](https://github.com/Hanagumori-B/VisionBOP/blob/master/ORIGINAL_README.md).**

This version has made some changes to the workspace logic of the original software, making it incompatible with loading the original workspace.

This version supports annotating images using ply format model files. In this version, a workspace allows loading only one image and multiple model files, and a dataset should contain multiple workspaces.

## MODIFICATION

- The functions of drawing and loading masks was blocked.
- Added models and related functions for the background plane.
- Added function for exporting BOP format datasets.
- Modified the loading and saving contents of the workspace, and added a quick switching function.
- Added a tool for viewing depth maps.

## License

GNU General Public License v3.0
