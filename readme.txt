Please install the required Python modules before running the code.

Information on the required Python modules is provided within each script (approximately within the first 7 lines). The tested package versions are listed below:

numpy                 1.26.4
matplotlib            3.10.8
pandas                3.0.1
scipy                 1.17.1
ImageIO               2.37.2
pillow                12.1.1

For each Python script, please use the attached file "LER test.txt" as the input and compare your output with the [Results] shown below.


1) Height Threshold

load_txt: specify the input path
threshold_ratio: t = 0.5
output_name: specify the output image file name

A binary image will be saved. If you do not change the file name, the default output file name will be 'test_threshold_0.50.png'.


2) ERF

TXT_PATH: specify the input path
OUT_BIN: specify the output path

A binary image will be saved to the specified output path.


3) PSD

png_path: input path (binary image PNG extracted from Height Threshold or ERF)

"LER_3sigma_nm" obtained directly from the edge positions was used as the representative value.
If LER is to be calculated from PSD integration, the integration range should be chosen appropriately.



