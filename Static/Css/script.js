<script>
document.addEventListener('DOMContentLoaded', function(){
  const form = document.getElementById('diseaseForm');
  const fileInput = document.getElementById('fileInput');
  const uploadArea = document.getElementById('uploadArea');
  const uploadPlaceholder = document.getElementById('uploadPlaceholder');
  const imagePreview = document.getElementById('imagePreview');
  const previewImage = document.getElementById('previewImage');
  const fileName = document.getElementById('fileName');
  const removeImage = document.getElementById('removeImage');
  const detectBtn = document.getElementById('detectBtn');
  const spinner = document.getElementById('diseaseSpinner');
  const ajaxResult = document.getElementById('ajaxResult');
  const predictionText = document.getElementById('predictionText');
  const resultMessage = document.getElementById('resultMessage');
  const confidenceBar = document.getElementById('confidenceBar');
  const confidenceText = document.getElementById('confidenceText');
  const resultImage = document.getElementById('resultImage');
  const analyzedImage = document.getElementById('analyzedImage');

  if (!form) {
    console.error('Form not found');
    return;
  }

  // File input change handler
  fileInput.addEventListener('change', function(e) {
    handleFile(e.target.files[0]);
  });

  // Drag and drop handlers
  uploadArea.addEventListener('dragover', function(e) {
    e.preventDefault();
    uploadArea.classList.add('border-green-500', 'bg-green-50');
  });

  uploadArea.addEventListener('dragleave', function(e) {
    e.preventDefault();
    uploadArea.classList.remove('border-green-500', 'bg-green-50');
  });

  uploadArea.addEventListener('drop', function(e) {
    e.preventDefault();
    uploadArea.classList.remove('border-green-500', 'bg-green-50');
    handleFile(e.dataTransfer.files[0]);
  });

  // Remove image handler
  removeImage.addEventListener('click', function() {
    fileInput.value = '';
    uploadPlaceholder.classList.remove('hidden');
    imagePreview.classList.add('hidden');
    detectBtn.disabled = true;
    ajaxResult.classList.add('hidden');
  });

  function handleFile(file) {
    if (!file) return;
    
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
      alert('Unsupported file type. Please upload a PNG or JPG image.');
      return;
    }
    
    if (file.size > 16 * 1024 * 1024) {
      alert('File too large. Max 16MB allowed.');
      return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
      previewImage.src = e.target.result;
      fileName.textContent = file.name;
      uploadPlaceholder.classList.add('hidden');
      imagePreview.classList.remove('hidden');
      detectBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // Form submission handler
  form.addEventListener('submit', function(e){
    e.preventDefault();

    if(!fileInput.files.length){
      alert('Please upload an image first.');
      return;
    }

    // Show loading state
    spinner.classList.remove('hidden');
    detectBtn.disabled = true;
    ajaxResult.classList.remove('hidden');
    predictionText.textContent = 'Analyzing image...';
    resultMessage.textContent = 'Processing your image with AI model';
    confidenceBar.style.width = '0%';
    confidenceText.textContent = '0%';
    resultImage.classList.add('hidden');

    const formData = new FormData(form);

    console.log('📤 Sending AJAX request...');

    // Add header to indicate we want JSON response
    fetch(form.action, {
      method: 'POST',
      body: formData,
      headers: {
        'X-Requested-With': 'XMLHttpRequest'
      }
    })
    .then(response => {
      console.log('📥 Response received:', response.status, response.statusText);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json().then(data => {
        console.log('📊 Response data:', data);
        return data;
      });
    })
    .then(data => {
      console.log('✅ Request successful:', data);
      
      // Hide spinner
      spinner.classList.add('hidden');
      detectBtn.disabled = false;

      if (data.error) {
        predictionText.textContent = 'Error';
        resultMessage.textContent = data.error;
        confidenceBar.style.width = '0%';
        confidenceText.textContent = '0%';
        return;
      }

      // Show results
      predictionText.textContent = data.prediction || 'Unknown';
      resultMessage.textContent = data.message || 'Analysis complete';
      
      // Update confidence bar
      const confidence = (data.confidence || 0.85) * 100;
      confidenceBar.style.width = confidence + '%';
      confidenceText.textContent = confidence.toFixed(1) + '%';
      
      // Show image if available
      if (data.image_url) {
        analyzedImage.src = data.image_url;
        resultImage.classList.remove('hidden');
      }
    })
    .catch(error => {
      console.error('❌ Request failed:', error);
      spinner.classList.add('hidden');
      detectBtn.disabled = false;
      predictionText.textContent = 'Error';
      resultMessage.textContent = 'Failed to analyze image. Please check console for details.';
      confidenceBar.style.width = '0%';
      confidenceText.textContent = '0%';
    });
  });
});
</script>