#############################################################################
# This api is contructed with Flask and FlaskRestFull

# Flask-RESTful is an extension for Flask that adds support for quickly building REST APIs
#############################################################################


from flask import Flask, jsonify
from flask_restful import reqparse, Api, Resource
import pickle

# Initialize the api
app = Flask(__name__)
api = Api(app)

# Reading the serialized model
model = pickle.load(open('point_11/iris_app/train_model/iris_model.pkl', 'rb'))

# Reading the parameters passed to the api as query string
# Only need two features as input data to execute the prediction
parser = reqparse.RequestParser()
parser.add_argument('feature_1', required=False, location=['values'])
parser.add_argument('feature_2', required=False, location=['values'])

class requestIris(Resource):
    """
    Class defined as resource in the api
    - Read the params and compute the predictions
    - Return the prediction as json object
    """

    def get(self):
        args = parser.parse_args()
        print(args)
        comp_1= float(args['feature_1'])
        comp_2 = float(args["feature_2"])
        prediction = (model.predict([[comp_1, comp_2]])).tolist()
        if(prediction[0] == 0):
            type="Iris-setosa"
        elif(prediction[0] == 1):
            type="Iris-versicolor"
        else:
            type="Iris-virginica"
        serve_model = {"Iris Type":type}
        print(serve_model)
        return jsonify(serve_model)
    
# For requests the endpoint to this api is: http://127.0.0.1:5000/classify
api.add_resource(requestIris, '/classify')
    
if __name__ == '__main__':
   app.run(debug=True)