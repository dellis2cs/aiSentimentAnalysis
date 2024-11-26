import React, { useState } from "react";
import "./App.css";

function App() {
  const [review, setReview] = useState("");
  const [sentiment, setSentiment] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent page reload
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ review }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();
      setSentiment(data.sentiment); // Update sentiment state
    } catch (error) {
      console.error("Error sending request:", error);
      setSentiment("Error analyzing sentiment");
    }
  };

  return (
    <div className="App p-4">
      <h1>Sentiment Analysis</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={review}
          onChange={(e) => setReview(e.target.value)}
          placeholder="Enter your review here..."
          rows="5"
          cols="50"
          className="p-4"
        ></textarea>
        <br />
        <button type="submit">Analyze Sentiment</button>
      </form>
      {sentiment && <h2>Sentiment: {sentiment}</h2>}
    </div>
  );
}

export default App;
