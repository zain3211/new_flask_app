let dropZone = document.getElementById('drop_zone');
let fileInput = document.getElementById('file');
let form = document.getElementById('upload_form');

// Handle the click event on the drop zone to trigger the file input click
dropZone.addEventListener('click', function() {
    fileInput.click();
});

// Handle the dragover event to prevent browser default behavior
dropZone.addEventListener('dragover', function(e) {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = 'copy';
});

// Handle the drop event to get the dropped files and set them to the file input
dropZone.addEventListener('drop', function(e) {
    e.preventDefault();
    e.stopPropagation();
    let files = e.dataTransfer.files;
    fileInput.files = files;
});

// Handle the change event on the file input to show a confirmation message
fileInput.addEventListener('change', function() {
    let file = this.files[0];
    if (file) {
        let message = document.createElement('p');
        message.textContent = "Image uploaded successfully!";
        form.insertBefore(message, form.firstChild);
    }
});