from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from diffusers import StableDiffusionPipeline
import torch
import os
from uuid import uuid4
from django.conf import settings

# Load the model only once (global)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Ensure MEDIA_ROOT exists
os.makedirs(settings.MEDIA_ROOT, exist_ok=True)


def index(request):
    return render(request, 'index.html')


class GenerateImageView(APIView):

    def post(self, request):
        prompt = request.data.get('prompt')
        if not prompt:
            return Response({"error": "Prompt is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Generate image
            image = pipe(prompt).images[0]

            # Create unique filename
            filename = f"{uuid4().hex}.png"
            path = os.path.join(settings.MEDIA_ROOT, filename)

            # Save image
            image.save(path)

            # Build full URL to access the image
            image_url = request.build_absolute_uri(settings.MEDIA_URL + filename)
            return Response({"image_url": image_url})

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
