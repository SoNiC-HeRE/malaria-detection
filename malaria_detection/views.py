from django.shortcuts import render
from .forms import ImageUploadForm
from .detecting_malaria import detect_malaria_parasites


def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            detected_class, parasitized_prob = detect_malaria_parasites(image)
            return render(request, 'malaria_detection/result.html', {'detected_class': detected_class, 'parasitized_prob': parasitized_prob})
    else:
        form = ImageUploadForm()
    return render(request, 'malaria_detection/index.html', {'form': form})