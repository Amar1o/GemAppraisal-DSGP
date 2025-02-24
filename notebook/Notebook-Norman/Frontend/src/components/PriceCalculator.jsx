import { useState, useEffect } from "react";
import axios from "axios";
import FileUpload from "./FileUpload.jsx";

// Dropdown component
function Dropdown({ label, name, options, value, onChange }) {
  return (
    <div className="form-control w-full">
      <label className="block font-medium text-gray-700">{label}:</label>
      <div className="mt-2">
        <select
          name={name}
          value={value}
          onChange={onChange}
          className="shadow rounded w-full py-2 px-3 text-gray-700 focus:outline-none focus:shadow-outline"
        >
          <option value="">Select {label}</option>
          {Object.entries(options).map(([key]) => (
            <option key={key} value={key}>
              {key}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

// Input Box component
function InputText({ label, name, value, onChange }) {
  return (
    <div className="form-control w-full">
      <label className="block font-medium text-gray-700">{label}:</label>
      <div className="mt-2">
        <input
          name={name}
          type="text"
          value={value}
          onChange={onChange}
          className="shadow rounded w-full py-2 px-3 text-gray-700 focus:outline-none focus:shadow-outline"
        />
      </div>
    </div>
  );
}

const PriceCalculator = () => {
  const [jsonData, setJsonData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [file, setFile] = useState(null);
  const [itemID, setItemID] = useState("");
  const [selectedValues, setSelectedValues] = useState({
    carat: "",
    cutQuality: "",
    shape: "",
    origin: "",
    color: "",
    colorIntensity: "",
    clarity: "",
    treatment: "",
    cut: "",
    type: "",
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("/encoded_features.json");
        if (!response.ok) throw new Error("Failed to load config");
        setJsonData(await response.json());
      } catch (error) {
        console.error("Error fetching JSON:", error);
        setError("Failed to load application configuration");
      }
    };
    fetchData();
  }, []);

  // Map backend keys to frontend state
  const mapBackendKeysToFrontend = (backendData) => {
  return {
    color: backendData["Color"] || "",
    shape: backendData["Shape"] || "",
    clarity: backendData["Clarity"] || "",
    cut: backendData["Cut"] || "",
    colorIntensity: backendData["Color Intensity"] || "",
    };
  };

  const handleFileUpload = async () => {
    if (!file) {
      setError("Please upload a video file.");
      return;
    }

    setLoading(true);
    setError("");
    const formData = new FormData();
    formData.append("file", file.file);

    try {
      const response = await axios.post("http://127.0.0.1:5000/price-calculator", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (response.data.ItemID) {
        setItemID(response.data.ItemID);
        setError("");
      } else {
        setError("Failed to extract ItemID.");
      }
    } catch (err) {
      setError("Error extracting ItemID. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleAutoFill = async () => {
    if (!itemID) {
      setError("Please upload a video first.");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.get(`http://127.0.0.1:5000/get-attributes?itemID=${itemID}`);
      if (response.data) {
        console.log("Backend Response:", response.data);
        setSelectedValues((prev) => ({
          ...prev,
          ...mapBackendKeysToFrontend(response.data),
        }));
      } else {
        setError("Failed to fetch attributes.");
      }
    } catch (err) {
      setError("Error fetching attributes. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setSelectedValues((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const validateForm = () => {
    return Object.values(selectedValues).every((val) => val);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    if (!validateForm()) {
      setError("Please fill in all required fields.");
      setLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append("ItemID", itemID);
    Object.entries(selectedValues).forEach(([key, value]) => {
      formData.append(key, value);
    });

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPrediction(response.data.price);
      console.log("Predicted Price:", response.data.price);
    } catch (err) {
      setError("Error predicting price. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedValues({
      carat: "",
      cutQuality: "",
      shape: "",
      origin: "",
      color: "",
      colorIntensity: "",
      clarity: "",
      treatment: "",
      cut: "",
      type: "",
    });
    setFile(null);
    setItemID("");
    setError("");
    setPrediction(null);
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(price);
  };

  if (!jsonData) return <p>Loading...</p>;

  return (
    <div>
      <form onSubmit={handleSubmit} className="w-full max-w-5xl mx-auto flex flex-col md:flex-row gap-10 mt-12 items-center md:items-start px-4">
        <div className="w-3/4">
          <FileUpload file={file} setFile={setFile} />
          <button type="button" onClick={handleFileUpload} className="mt-4 bg-gray-700 hover:bg-gray-500 text-white font-semibold py-2 px-4 rounded">
            Connect
          </button>
        </div>

        <div>
          <button type="button" onClick={handleAutoFill} className="mb-4 bg-blue-500 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded">
            Auto-Fill Fields
          </button>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-y-5 gap-x-10 w-full min-w-[390px] md:min-w-[500px]">
            <Dropdown label="Type" name="type" options={jsonData.type_mapping} value={selectedValues.type} onChange={handleChange} />
            <InputText label="Carat" name="carat" value={selectedValues.carat} onChange={handleChange} />
            <Dropdown label="Color" name="color" options={jsonData.color_mapping} value={selectedValues.color} onChange={handleChange} />
            <Dropdown label="Color Intensity" name="colorIntensity" options={jsonData.color_intensity_mapping} value={selectedValues.colorIntensity} onChange={handleChange} />
            <Dropdown label="Shape" name="shape" options={jsonData.shape_mapping} value={selectedValues.shape} onChange={handleChange} />
            <Dropdown label="Clarity" name="clarity" options={jsonData.clarity_mapping} value={selectedValues.clarity} onChange={handleChange} />
            <Dropdown label="Cut" name="cut" options={jsonData.cut_mapping} value={selectedValues.cut} onChange={handleChange} />
            <Dropdown label="Cut Quality" name="cutQuality" options={jsonData.cut_quality_mapping} value={selectedValues.cutQuality} onChange={handleChange} />
            <Dropdown label="Origin" name="origin" options={jsonData.origin_mapping} value={selectedValues.origin} onChange={handleChange} />
            <Dropdown label="Treatment" name="treatment" options={jsonData.treatment_mapping} value={selectedValues.treatment} onChange={handleChange} />
          </div>

          <button type="submit" className="bg-gray-700 hover:bg-gray-500 text-white py-2 px-4 rounded mt-5">Submit</button>
          <button type="button" onClick={handleClear} className="ml-4 bg-gray-300 text-gray-700 py-2 px-4 rounded">Clear</button>
          {prediction !== null && (
            <p className="mt-6 text-xl font-bold text-gray-800">
              Estimated Price: {formatPrice(prediction)}
            </p>
          )}
        </div>
      </form>
    </div>
  );
};

export default PriceCalculator;
