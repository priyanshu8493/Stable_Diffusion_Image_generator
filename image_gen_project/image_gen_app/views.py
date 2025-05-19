from django.shortcuts import render

# Create your views here.

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from diffusers import StableDiffusionPipeline
import torch
import os
from uuid import uuid4
from django.conf import settings


def index(request):
    return render(request, 'index.html')


class GenerateImageView(APIView):

    def post(self, request):

        prompt = request.data.get('prompt')
        if not prompt:
            return Response({"error": "Prompt is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Load the model (tip: move this to __init__ or cache for performance)
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")

        # Generate image
        image = pipe(prompt).images[0]

        # Save image
        filename = f"{uuid4().hex}.png"
        path = os.path.join(settings.MEDIA_ROOT, filename)
        image.save(path)

        # Return URL
        image_url = request.build_absolute_uri(settings.MEDIA_URL + filename)
        return Response({"image_url": image_url})
