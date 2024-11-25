function showLoading(message) {
    document.getElementById('loading-overlay').style.display = 'flex';
    document.getElementById('loading').textContent = message;
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

function updateSliderValue() {
    const slider = document.getElementById('slider');
    const sliderValue = document.getElementById('slider-value');
    sliderValue.textContent = slider.value;
}

fetch('/returnmodelslist')
    .then(response => response.json())
    .then(data => {
      const modelSelect = document.getElementById('modelSelect');
      console.log(data)
      // Populate dropdown with options
      data.result.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.text = model;
        modelSelect.add(option);
      });
      // Automatically select the first model
      if (data.result.length > 0) {
        modelSelect.value = data.result[0];
        loadModel()
      }
    })
    .catch(error => console.error('Error fetching models:', error));

function loadModel() {
    showLoading('The model is loading...');
    const modelSelect = document.getElementById('modelSelect');
    const selectedModel = modelSelect.value;

    if (selectedModel) {
        // Fetch model data from /loadmodel using the selected model
        fetch(`/loadmodel?model=${selectedModel}`,{
            method: 'POST'})
            .then(response => response.json())
            .then(data => {
                // Handle the loaded model data
                console.log('Loaded Model:', data);
                hideLoading();
            })
            .catch(error => {
                console.error('Error loading model:', error);
                hideLoading();
            });
    } else {
        console.log('Please select a model.');
    }
}

function updateResultText(texts) {
    const resultText = document.getElementById('result-text');

    // Clear previous content
    resultText.innerHTML = '';

    // Add each text to the resultText
    texts.forEach(text => {
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        resultText.appendChild(paragraph);
    });
}

function showResultImage(imageUrls) {
    const resultImagesContainer = document.getElementById('result-images');

    // Clear previous content
    resultImagesContainer.innerHTML = '';

    // Add each image to the resultImagesContainer in a grid
    imageUrls.forEach(imageUrl => {
        const image = document.createElement('img');
        image.src = imageUrl;
        resultImagesContainer.appendChild(image);
    });

    // Display the container
    resultImagesContainer.style.display = 'grid';
}

function hideResultImage() {
    const resultImagesContainer = document.getElementById('result-images');
    resultImagesContainer.innerHTML = '';
    resultImagesContainer.style.display = 'none';
}

function dragOverHandler(event) {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
}

function dropHandler(event) {
    event.preventDefault();
    const files = event.dataTransfer.files;

    if (files.length > 0) {
        handleFiles(files);
    }
}

function clickFileInput() {
    document.getElementById('file-input').click();
}

function handleFile() {
    const input = document.getElementById('file-input');
    const files = input.files;

    if (files.length > 0) {
        handleFiles(files);
    }
}

function handleFiles(files) {
    showLoading('Loading the prediction of the text from the image...');

    for (const image of files) {
        // Check if the file is an image
        if (!image.type.startsWith('image/')) {
            alert('Please upload only image files.');
            return;
        }

        // Assuming you have an API endpoint '/imageprediction' to send the image file
        // You can use fetch to send the image file to the server
        const imageApiUrl = '/textprediction';
        const imageFormData = new FormData();
        const sliderValue = document.getElementById('slider').value;

        imageFormData.append('image', image);
        imageFormData.append('n', sliderValue);

        fetch(imageApiUrl, {
            method: 'POST',
            body: imageFormData
        })
            .then(response => response.json())
            .then(data => {
                // Handle the response from the image prediction server if needed
                console.log(data);
                hideLoading();
                // Display the result on the page
                updateResultText(data.result);
                hideResultImage();
            })
            .catch(error => {
                console.error('Image Prediction Error:', error);
                hideLoading();
            });
    }
}

function submitText() {
    const textInput = document.getElementById('text-input').value;
    const sliderValue = document.getElementById('slider').value;

    showLoading('Loading the prediction of the image from the text...');

    // Assuming you have an API endpoint '/textprediction' to send the text
    // You can use fetch to send the text to the server
    const textApiUrl = '/imageprediction';
    const textFormData = new FormData();
    textFormData.append('text', textInput);
    textFormData.append('n', sliderValue);

    fetch(textApiUrl, {
        method: 'POST',
        body: textFormData
    })
        .then(response => response.json())
        .then(data => {
        // Handle the response from the text prediction server if needed
        console.log(data);
        hideLoading();
        // Display the result on the page

        updateResultText(['']);
        showResultImage(data.result);
    })
        .catch(error => {
            console.error('Text Prediction Error:', error);
            hideLoading();
        });
}