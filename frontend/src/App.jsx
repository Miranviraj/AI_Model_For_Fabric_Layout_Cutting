import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import jsPDF from 'jspdf';
import spinner from './assets/sprinner.svg';  


function App() {
  const patternSizes = {
    Small: { width: 40, height: 40 },
    Medium: { width: 50, height: 50 },
    Large: { width: 60, height: 60 },
    XL: { width: 70, height: 70 },
    XXL: { width: 90, height: 90 },
  };

  const fabricOptions = [
    { label: '70 x 70', width: 70, height: 70 },
    { label: '80 x 80', width: 80, height: 80 },
    { label: '100 x 100', width: 100, height: 100 },
    { label: '150 x 150', width: 150, height: 150 },
  ];

  const [selectedFabric, setSelectedFabric] = useState(fabricOptions[0]);
  const [patternEntries, setPatternEntries] = useState([{ sizeLabel: 'Small', quantity: 1 }]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef(null);

  const handleSubmit = async () => {
    const patterns = [];
    patternEntries.forEach(entry => {
      const size = patternSizes[entry.sizeLabel];
      for (let i = 0; i < entry.quantity; i++) {
        patterns.push({ width: size.width, height: size.height });
      }
    });

    setLoading(true);
    try {
      const res = await axios.post('http://localhost:5000/api/hybrid-layout', {
        fabric: {
          width: selectedFabric.width,
          height: selectedFabric.height
        },
        patterns
      });
      if (res.data.unplaced_patterns > 0) {
  alert(`⚠️ ${res.data.unplaced_patterns} patterns could not be placed in selected fabric.`);
}

      setResult(res.data);
    } catch (err) {
      console.error("API error:", err);
      alert("❌ Failed to connect to AI server.");
    } finally {
      setLoading(false);
    }
  };

  const exportToPDF = async () => {
    const canvas = canvasRef.current;
    const imgData = canvas.toDataURL("image/png");
    const pdf = new jsPDF();
    const pdfWidth = 180;
    const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
    pdf.addImage(imgData, 'PNG', 10, 10, pdfWidth, pdfHeight);
    pdf.save("fabric_layout.pdf");
  };

 const doesFabricFit = () => {
  const totalPatternArea = patternEntries.reduce((sum, entry) => {
    const size = patternSizes[entry.sizeLabel];
    const quantity = entry.quantity || 1;
    return sum + size.width * size.height * quantity;
  }, 0);

  const fabricArea = selectedFabric.width * selectedFabric.height;
  return totalPatternArea <= fabricArea;
};



  useEffect(() => {
    if (result?.layout && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const scale = 4;
      canvas.width = selectedFabric.width * scale;
      canvas.height = selectedFabric.height * scale;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#f9fafb';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const colors = ['#4caf50', '#2196f3', '#ff9800', '#e91e63', '#9c27b0'];
result.layout.forEach((rect, i) => {
  const x = rect.x * scale;
  const y = rect.y * scale;
  const w = rect.width * scale;
  const h = rect.height * scale;

  ctx.fillStyle = colors[i % colors.length];
  ctx.fillRect(x, y, w, h);

  ctx.strokeStyle = '#000';
  ctx.strokeRect(x, y, w, h);

// Find size label from dimensions
  const sizeLabel = Object.entries(patternSizes).find(
    ([_, dims]) => dims.width === rect.width && dims.height === rect.height
  )?.[0] || "Unknown";

  ctx.fillStyle = '#000';
  ctx.font = '10px Arial';
  ctx.textAlign = 'center';
  ctx.fillText(`${sizeLabel}${rect.rotated ? ' ↻' : ''}`, x + w / 2, y + h / 2);
});

    }
  }, [result, selectedFabric]);


  
  return (
    <div className="min-h-screen p-8  bg-gradient-to-b from-[#efecea] via-[#f1e0ca] to-[#f7dbb7] font-sans">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 text-gray-800 animate-bounce">🧠 AI Fabric Cutting Optimizer</h1>

        <div className="bg-gradient-to-l from-[#bbaa9b] via-[#ddcbb5] to-[#dcd1c3] p-6 rounded-lg shadow-lg mb-6 w-2xl">
          <h2 className="text-xl font-semibold text-gray-700 mb-4">🪡 Fabric & Patterns</h2>

          <label className="block mb-2 text-gray-600 font-medium">Select Fabric Size:</label>
          <select
            className="border border-gray-300 rounded px-3 py-2  mb-4 bg-white shadow-sm w-20px"
            value={selectedFabric.label}
            onChange={(e) => {
              const fabric = fabricOptions.find(f => f.label === e.target.value);
              setSelectedFabric(fabric);
            }}
          >
            {fabricOptions.map((f, i) => (
              <option key={i} value={f.label}>{f.label}</option>
            ))}
          </select>

          <div className="space-y-10">
            {patternEntries.map((entry, i) => (
              <div key={i} className="flex gap-4 items-center">
                <select
                  className="border border-gray-300 rounded px-3 py-2 bg-white"
                  value={entry.sizeLabel}
                  onChange={(e) => {
                    const updated = [...patternEntries];
                    updated[i].sizeLabel = e.target.value;
                    setPatternEntries(updated);
                  }}
                >
                  {Object.keys(patternSizes).map(size => (
                    <option key={size} value={size}>{size}</option>
                  ))}
                </select>

                <input
                  type="number"
                  className="border px-3 py-2 w-20 rounded border-gray-300"
                  value={entry.quantity}
                  onChange={(e) => {
                    const updated = [...patternEntries];
                    updated[i].quantity = Number(e.target.value);
                    setPatternEntries(updated);
                  }}
                  min={1}
                />

                <button
                  onClick={() => {
                    const updated = [...patternEntries];
                    updated.splice(i, 1);
                    setPatternEntries(updated);
                  }}
                  className="bg-red-500 text-white px-2 rounded shadow hover:bg-red-600"
                >✖</button>
              </div>
            ))}

            <button
              className="mt-2 bg-green-600 text-white px-4 py-1 rounded shadow hover:bg-green-700"
              onClick={() => setPatternEntries([...patternEntries, { sizeLabel: 'Small', quantity: 1 }])}
            >➕ Add Pattern</button>
          </div>
        </div>

        {loading ? (
          <div className="flex justify-center mt-4">
    <img src={spinner} alt="loading" className="w-10 h-10 animate-spin" />
  </div>
        ) : (
          <button
            onClick={handleSubmit}
            className="bg-blue-600 text-white px-5 py-2 rounded shadow hover:bg-blue-700 transition-all"
          >🚀 Submit to AI</button>
        )}

        {result && (
          <div className="mt-10 flex flex-col md:flex-row gap-6">
            <div className="bg-gradient-to-l from-[#bbaa9b] via-[#ddcbb5] to-[#dcd1c3] p-4 rounded shadow w-full md:w-1/3">
              <h2 className="text-lg font-semibold mb-2">📋 Summary</h2>
              <ul className="space-y-2 text-gray-700">

                <li>🟢 Used Area: {result.used_area}</li>
                <li>🔴 Waste Area: {result.waste_area}</li>
                <li>⚠️ Waste %: {result.waste_percentage}%</li>
                <li>📦 Total Patterns: {patternEntries.reduce((sum, e) => sum + e.quantity, 0)}</li>
                <li>🧩 Fabric Size: {selectedFabric.label}</li>


                {result.possible_cuts && (
  <div className="mt-4 text-sm text-gray-700">
    <h3 className="font-semibold mb-1">♻️ From Waste, You Can Cut:</h3>
    <ul className="list-disc ml-4">
      {Object.entries(result.possible_cuts).map(([key, val]) => (
        <li key={key}>
          {val} × {key}
        </li>
      ))}
    </ul>
   {result.unplaced_patterns > 0 && (
  <p className="text-red-600 mt-2 text-sm">
    ⚠️ {result.unplaced_patterns} pattern(s) could not be placed in selected fabric.
  </p>
)}


  </div>
)}
              </ul>
              <button
                onClick={exportToPDF}
                className="mt-4 bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700"
              >📄 Export PDF</button>

            

            </div>

            <div className="w-full md:w-2/3 border rounded bg-gradient-to-l from-[#bbaa9b] via-[#ddcbb5] to-[#dcd1c3] p-4 shadow">
              <canvas ref={canvasRef} className="block mx-auto border shadow-md" />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;