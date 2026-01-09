async function diagnose() {
    const textInput = document.getElementById("textInput").value;
    const imageInput = document.getElementById("imageInput").files[0];
    const outputText = document.getElementById("reversedText");
    const outputImage = document.getElementById("outputImage");

    let data = { q: textInput };

    if (imageInput) {
        const reader = new FileReader();
        reader.onload = async () => {
            const base64 = reader.result.split(',')[1];
            data.img = base64;

            try {
                const res = await fetch("http://127.0.0.1:5000/submit", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data)
                });

                const responseData = await res.json();

                if (responseData.final_prediction) {
                    outputText.innerText =
                        "Detected Disease: " + responseData.final_prediction;
                } else {
                    outputText.innerText = "Prediction failed.";
                }

                outputImage.style.display = "none";

            } catch (err) {
                console.error("Submit error:", err);
            }
        };
        reader.readAsDataURL(imageInput);
    } else {
        try {
            const res = await fetch("http://127.0.0.1:5000/submit", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });

            const responseData = await res.json();

            if (responseData.final_prediction) {
                outputText.innerText =
                    "Detected Disease: " + responseData.final_prediction;
            } else {
                outputText.innerText = "Please upload image or enter text.";
            }

        } catch (err) {
            console.error("Submit error:", err);
        }
    }
}
