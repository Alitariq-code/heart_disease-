from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from heart.first.api1 import load_models_and_vote
from heart.second.api import load_models_and_votes

class Pre1(APIView):
    def post(self, request, *args, **kwargs):
        # Extract data from the incoming JSON request
        received_data = request.data

        # Define the expected features in the correct order
        expected_features = ['age', 'prevalentHyp', 'diabetes', 'sysBP', 'diaBP', 'glucose']

        # Initialize an empty list to hold the input features
        input_features = []

        # Iterate over the expected features and extract them from received_data
        for feature in expected_features:
            # Check if each expected feature is present and is an integer (or can be converted to an integer)
            if feature in received_data:
                try:
                    # Convert to integer and append to input_features list
                    input_features.append(int(received_data[feature]))
                except ValueError:
                    # If conversion to integer fails, return an error response
                    return Response({'error': f'Invalid value for {feature}. Must be an integer.'}, status=status.HTTP_400_BAD_REQUEST)
            else:
                # If any expected feature is missing, return an error response
                return Response({'error': f'Missing feature: {feature}'}, status=status.HTTP_400_BAD_REQUEST)

        # Now input_features contains the ordered list of integer features to pass to the model
        prediction = load_models_and_vote(input_features)
        final_prediction = False if prediction[0] == 0 else True
        

        # Prepare and send the response, including the prediction
        response_data = {
            'message': 'Data received and processed successfully.',
            'receivedData': received_data,  # Echo back the received data
            'prediction': final_prediction  # Include the model's prediction
        }

        return Response(response_data, status=status.HTTP_201_CREATED)
    
    
    
    
    
class Pre2(APIView):
    def post(self, request, *args, **kwargs):
        # Extract data from the incoming JSON request
        received_data = request.data

        # Define the expected features in the correct order
        expected_features = ['age', 'prevalentHyp', 'diabetes', 'sysBP', 'diaBP', 'glucose']

        # Initialize an empty list to hold the input features
        input_features = []

        # Iterate over the expected features and extract them from received_data
        for feature in expected_features:
            # Check if each expected feature is present and is an integer (or can be converted to an integer)
            if feature in received_data:
                try:
                    # Convert to integer and append to input_features list
                    input_features.append(int(received_data[feature]))
                except ValueError:
                    # If conversion to integer fails, return an error response
                    return Response({'error': f'Invalid value for {feature}. Must be an integer.'}, status=status.HTTP_400_BAD_REQUEST)
            else:
                # If any expected feature is missing, return an error response
                return Response({'error': f'Missing feature: {feature}'}, status=status.HTTP_400_BAD_REQUEST)

        # Now input_features contains the ordered list of integer features to pass to the model
        prediction = load_models_and_vote(input_features)
        final_prediction = False if prediction[0] == 0 else True
        

        # Prepare and send the response, including the prediction
        response_data = {
            'message': 'Data received and processed successfully.',
            'receivedData': received_data,  # Echo back the received data
            'prediction': final_prediction  # Include the model's prediction
        }

        return Response(response_data, status=status.HTTP_201_CREATED)
    
    
    
    