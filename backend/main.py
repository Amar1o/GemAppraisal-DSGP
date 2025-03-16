from flask import Flask
from flask_cors import CORS
from videoModel import video_routes
from pricePrediction import prediction_routes
from pictureModel import picture_routes
from ClassifyModel import image_routes

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Register Blueprints
app.register_blueprint(video_routes, url_prefix="/video")
app.register_blueprint(prediction_routes, url_prefix="/predict")
app.register_blueprint(picture_routes,url_prefix="/picture")
app.register_blueprint(image_routes,url_prefix="/image")


if __name__ == "__main__":
    app.run(debug=True)