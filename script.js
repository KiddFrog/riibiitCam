document.addEventListener("DOMContentLoaded", function() {
    // Get the reference to the image container
    var imageContainer = document.getElementById("imageContainer");
  
    // Fetch images from the "PICTURES" folder
    fetchImages();
  
    function fetchImages() {
      // Assume the images are in the "PICTURES" folder
      var folderPath = "PICTURES/";
  
      // Use a fetch request to get the list of images
      fetch(folderPath)
        .then(response => response.text())
        .then(data => {
          // Parse the HTML response to extract image filenames
          var parser = new DOMParser();
          var doc = parser.parseFromString(data, "text/html");
          var images = doc.querySelectorAll("a[href$='.jpg']");
  
          // Iterate through the images and display them
          images.forEach(function(image) {
            var imageUrl = folderPath + image.getAttribute("href");
            displayImage(imageUrl);
          });
        })
        .catch(error => console.error("Error fetching images:", error));
    }
  
    function displayImage(imageUrl) {
      // Create an img element and set its source
      var imgElement = document.createElement("img");
      imgElement.src = imageUrl;
  
      // Append the img element to the image container
      imageContainer.appendChild(imgElement);
    }
  });
  