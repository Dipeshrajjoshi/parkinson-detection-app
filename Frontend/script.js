const BACKEND_URL = "http://127.0.0.1:5000";

// Create 22 inputs dynamically
window.onload = () => {
  const container = document.getElementById("features");
  for (let i = 0; i < 22; i++) {
    const input = document.createElement("input");
    input.type = "number";
    input.className = "input";
    input.placeholder = `Feature ${i + 1}`;
    container.appendChild(input);
  }
};

document.getElementById("predict-form").addEventListener("submit", async function (e) {
  e.preventDefault();
  const inputs = document.querySelectorAll(".input");
  const values = Array.from(inputs).map(i => i.value.trim());

  if (values.length < 26 || values.slice(4).some(v => isNaN(v) || v === "")) {
    alert("Please fill all fields with valid numbers.");
    return;
  }

  const name = values[0], age = values[1], address = values[2], doctor = values[3];
  const features = values.slice(4).map(Number);

  const res = await fetch(`${BACKEND_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features })
  });

  const data = await res.json();
  document.getElementById("result").innerHTML = `
    <h3>🧾 Diagnosis Report</h3>
    <p><strong>Patient:</strong> ${name}, Age ${age}</p>
    <p><strong>Address:</strong> ${address}</p>
    <p><strong>Doctor:</strong> Dr. ${doctor}</p>
    <p><strong>Prediction:</strong> ${data.result}</p>
    <p><strong>Date:</strong> ${new Date().toLocaleString()}</p>
  `;
});

function printReport() {
  window.print();
}

async function generateAndShowGraphs() {
  const res = await fetch(`${BACKEND_URL}/generate-plots`, { method: "POST" });
  const result = await res.json();
  if (result.status === "success") {
    document.getElementById("confusion").src = `${BACKEND_URL}/images/confusion.png`;
    document.getElementById("importance").src = `${BACKEND_URL}/images/importance.png`;
    document.getElementById("confusion").style.display = "block";
    document.getElementById("importance").style.display = "block";
  }
}
