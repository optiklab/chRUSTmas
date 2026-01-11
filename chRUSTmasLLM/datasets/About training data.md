# DATA and TASK

I am creating a program simple neural network following some of the examples in the internet, in Rust. The idea is to provide a general workflow and introduce the inner workings of a neural network. With this foundation, later I can expand and enhance the library to handle larger datasets, more complex network architectures, and additional features.

I am building a classifier that can identify images of cats and non-cats. To simplify the process we have converted cats images into 3-channel matrices of size 64 x 64, which represents the image resolution. If we flatten this matrix, we obtain a vector of size (3 x 64 x 64), which is equal to 12288. This is the number of input features (columns in CSV files + `y` column with `labels` which is basically `1` or `0` for cat and non-cat image) that will be fed into the network. Although we can make the network adaptable to different input feature sizes, first we will design it specifically for this dataset for now.

# TRAINING and TEST SETs

The typical process to convert cat images into CSV files with 12,288 features follows these steps:

1. Load the Image
- Read image file (JPG, PNG, etc.) using an image processing library
- Ensure it's in RGB color format (3 channels)

2. Resize to 64×64
- Resize/scale the image to exactly 64×64 pixels
- Use interpolation methods (bilinear, bicubic) to maintain quality

3. Extract Pixel Values
- Image is now a 3D array: [64, 64, 3] or [3, 64, 64] depending on format
- Each pixel has 3 values (Red, Green, Blue) ranging from 0-255

4. Normalize Pixel Values
- Divide all values by 255 to scale them to range [0, 1]
- Or use other normalization: (pixel - mean) / std_dev

5. Flatten the Array
- Reshape 3D array [3, 64, 64] into 1D vector of length 12,288
- Common flattening order: all red channel values, then green, then blue
- Alternative: interleaved (R,G,B of pixel 1, then pixel 2, etc.)

6. Create Label
- Assign label: 1 for cat, 0 for non-cat
- This becomes the "y" column

7. Write to CSV
- Each row: one image (12,288 feature columns + 1 label column)
- Column names: typically "pixel_0", "pixel_1", ..., "pixel_12287", "y"

Common Python approach (for reference when adjusting data):

```
    from PIL import Image
    import numpy as np

    img = Image.open('cat.jpg').convert('RGB')
    img = img.resize((64, 64))
    pixels = np.array(img) / 255.0  # Shape: (64, 64, 3)
    flattened = pixels.flatten()     # Shape: (12288,)
```

To adjust your data later, you'd reverse this: reshape 12,288 → (64, 64, 3), then modify and re-flatten.

Another example in C# for better understanding:
```
using System;
using System.IO;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

public class ImageToCSV
{
    public static float[] ConvertImageToFeatures(string imagePath)
    {
        // Load and resize image to 64x64
        using (var image = Image.Load<Rgb24>(imagePath))
        {
            image.Mutate(x => x.Resize(64, 64));
            
            // Create array to store flattened pixels
            float[] features = new float[64 * 64 * 3]; // 12,288
            int index = 0;
            
            // Extract and normalize pixel values
            for (int c = 0; c < 3; c++) // For each channel (R, G, B)
            {
                for (int y = 0; y < 64; y++)
                {
                    for (int x = 0; x < 64; x++)
                    {
                        var pixel = image[x, y];
                        float value = c switch
                        {
                            0 => pixel.R / 255.0f,
                            1 => pixel.G / 255.0f,
                            2 => pixel.B / 255.0f,
                            _ => 0
                        };
                        features[index++] = value;
                    }
                }
            }
            
            return features;
        }
    }
    
    public static void CreateCSVFromImages(string[] imagePaths, int[] labels, string outputPath)
    {
        using (var writer = new StreamWriter(outputPath))
        {
            // Write header
            var header = string.Join(",", 
                Enumerable.Range(0, 12288).Select(i => $"pixel_{i}")) + ",y";
            writer.WriteLine(header);
            
            // Process each image
            for (int i = 0; i < imagePaths.Length; i++)
            {
                var features = ConvertImageToFeatures(imagePaths[i]);
                var row = string.Join(",", features.Select(f => f.ToString("F6"))) 
                    + "," + labels[i];
                writer.WriteLine(row);
            }
        }
    }
}
```

Usage:
```
string[] catImages = Directory.GetFiles("cats/", "*.jpg");
string[] nonCatImages = Directory.GetFiles("non_cats/", "*.jpg");

var allImages = catImages.Concat(nonCatImages).ToArray();
var labels = Enumerable.Repeat(1, catImages.Length)
    .Concat(Enumerable.Repeat(0, nonCatImages.Length))
    .ToArray();

ImageToCSV.CreateCSVFromImages(allImages, labels, "dataset.csv");
```

# TRAINING DATA

Can you write a python script that will process CSV files #file:test_set.csv and #file:training_set.csv and convert images back from arrays of RGB to images?

Perfect! I've created a Python script csv_to_images.py that will:

- Read both CSV files (training_set.csv and test_set.csv)
- Extract the pixel data (columns 0-12287) and labels (column "y")
- Reshape the data from flat 12,288 values back to 3×64×64 (RGB channels)
- Transpose to 64×64×3 format for proper image display
- Scale values back to 0-255 range if needed
- Save as PNG images with names like:
-- cat_0000.png, cat_0001.png for cat images (y=1)
-- non_cat_0000.png, non_cat_0001.png for non-cat images (y=0)

To run the script:
```
pip install pandas numpy pillow

cd c:\usr\src\rust\chRUSTmasLLM2025\chRUSTmasLLM\chRUSTmasLLM\datasets
python csv_to_images.py
```


## TODO AY. Generalize it to make it adaptable to different input feature sizes.