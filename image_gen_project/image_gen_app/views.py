from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from diffusers import StableDiffusionPipeline
import torch
import io

# Load the model only once (global)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")


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

            # Write image to an in-memory buffer
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)

            # Return the PNG bytes directly
            response = HttpResponse(buffer.getvalue(), content_type="image/png")
            response["Content-Disposition"] = "inline; filename=generated.png"
            response["Cache-Control"] = "no-store"
            return response

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
