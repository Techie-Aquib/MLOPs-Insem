import joblib
from django.shortcuts import render
from django.http import JsonResponse
import numpy as np

model = joblib.load(r'C:\Users\student\exam\pythonProject1\mlops\classifier\model.joblib')


def predict(request):
    if request.method == 'POST':
        features = [
            float(request.POST.get('sepal_length')),
            float(request.POST.get('sepal_width')),
            float(request.POST.get('petal_length')),
            float(request.POST.get('petal_width')),
        ]

        prediction = model.predict([features])

        prediction_list = prediction.tolist() if hasattr(prediction, 'tolist') else prediction.tolist()

        return JsonResponse({'prediction': prediction_list})

    return render(request, 'home.html')
