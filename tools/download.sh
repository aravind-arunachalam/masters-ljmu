"""
Author: Aravind Kumar Arunachalam
Implementation for GQA dataset
Master's in Machine Learning and AI
"""

# All data downloaded from https://cs.stanford.edu/people/dorarad/gqa/download.html

# GQA image files. Large file and will take time to download
wget -P ../data/gqa https://nlp.stanford.edu/data/gqa/images.zip
unzip ../data/gqa/images.zip
rm ../data/gqa/images.zip

# Spatial features for images in the GQA dataset. Large file and will take time to download
wget -P ../data/gqa https://nlp.stanford.edu/data/gqa/spatialFeatures.zip
unzip ../data/gqa/spatialFeatures.zip
rm ../data/gqa/spatialFeatures.zip


# Object features for images in the GQA dataset. Large file and will take time to download
wget -P ../data/gqa https://nlp.stanford.edu/data/gqa/objectFeatures.zip
unzip ../data/gqa/objectFeatures.zip
rm ../data/gqa/objectFeatures.zip

# Question sets for GQA dataset. Large file and will take time to download
wget -P ../data/gqa https://nlp.stanford.edu/data/gqa/questions1.2.zip
unzip ../data/gqa/questions1.2.zip
rm ../data/gqa/questions1.2.zip

# Scene graphs for GQA dataset.
wget -P ../data/gqa https://nlp.stanford.edu/data/gqa/sceneGraphs.zip
unzip ../data/gqa/sceneGraphs.zip
rm ../data/gqa/sceneGraphs.zip