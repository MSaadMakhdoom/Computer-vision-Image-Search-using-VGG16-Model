import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    print("Image store Feature Path",feature_path)
    features.append(np.load(feature_path))
    print("Image Feature store  Load",np.load(feature_path))

    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))

    print("Image feature store Path",Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        im = Image.open(file.stream)  # PIL image

        img = im.convert("RGB")

        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        print("Upload Image Shape",query.shape)

        print("Uploade image ",query)


        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        print("Each Image Distance with respect to query image",dists)
        
        ids = np.argsort(dists)[:6]  # Top 6 results
        
        print("Sorted Image Distance with respect to query image",ids)
        
        scores = [(dists[id], img_paths[id]) for id in ids]

        print("Similarity Score:",scores)
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
