The dataset for this component is based on videos from [BARD dataset](https://github.com/GabrieleGiudic/BARD).

If you want to use our dataset, firstly you should download files dataset.csv and annotations.json from [yandex disk](https://disk.yandex.ru/d/mszFKl3780zRFQ) and put them in this folder.
Note that you would need selenium python library.
Afther that you should download videos using download_videos.py.
This script downloads all videos from BARD dataset, but we downloaded only first 1742 of them, 
because downloading is not really fast.

In the file annotations.json there are stored manual annotations for 495 random videos, 382 out of them are not invalid.