import { useState, useEffect } from "react";
import axios from "axios";
import FileUpload from "./FileUpload.jsx";

function Dropdown({ label, name, options, onChange }) {
  return (
    <div className="form-control w-full max-w-xs">
      <label className="form-control w-full max-w-xs block font-medium text-gray-700">{label}: </label>
      <div className="mt-2">
      <select name={name} onChange={onChange} className="select shadow rounded w-full py-2 px-3 text-gray-700 focus:outline-none focus:shadow-outline">
        <option value="" disabled></option>
        {Object.entries(options).map(([key, value]) => (
          <option key={key} value={value}>
            {key}
          </option>
        ))}
      </select>
      </div>
    </div>
  );
}  

function InputText({ label, name, onChange }) {
  return (
    <div className="form-control w-full max-w-xs">
      <label className="form-control w-full max-w-xs block font-medium text-gray-700">{label}: </label>
      <div className="mt-2">
      <input name={name} type='text' onChange={onChange} className="shadow rounded w-full py-2 px-3 text-gray-700 focus:outline-none focus:shadow-outline" />
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
        if (!response.ok) throw new Error('Failed to load config');
        setJsonData(await response.json());
      } catch (error) {
        console.error("Error fetching JSON:", error);
        setError("Failed to load application configuration");
      }
    };
    fetchData();
  }, []);
  const handleChange = (e) => {
    setSelectedValues({ ...selectedValues, [e.target.name]: e.target.value });
  };
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    const formData = new FormData();
  
    try {
      // Append all selected values to FormData
      Object.entries(selectedValues).forEach(([key, value]) => {
        if (value) {
          formData.append(key, value);
        }
      });
  
      // Append file if exists
      if (file) {
        formData.append("file", file.file);
      }
  
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data", // Let axios set this automatically
        },
      });
  
      setPrediction(response.data.price);
    } catch (err) {
      setError("Error predicting price. Please try again.");
      console.error("Prediction Error:", err.message);
      if (err.response) {
        console.error("Server Response:", err.response.data);
      }
    } finally {
      setLoading(false);
    }
  };
  

  const handleClear = (e) => {
    e.preventDefault();
    setSelectedValues({
      carat:"",
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
    setFile(null); // Reset file as well
  };

  if (!jsonData) {
    return <p>Loading...</p>;
  }
  

  return (
    <div>
      <form onSubmit={handleSubmit} className="w-full max-w-4xl mx-auto flex sm:flex-wrap md:flex-nowrap gap-10 mt-12 items-center">
      <FileUpload />
        <div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <InputText
        label="Carat" 
        name="carat"  
        onChange={handleChange} 
      />
      <Dropdown 
        label="Cut Quality" 
        name="cutQuality" 
        options={jsonData.cut_quality_mapping} 
        onChange={handleChange} 
      />
        <Dropdown
          label="Shape"
          name="shape"
          options={jsonData.shape_mapping}
          onChange={handleChange}
        />
        <Dropdown
          label="Origin"
          name="origin"
          options={jsonData.origin_mapping}
          onChange={handleChange}
        />
        <Dropdown
          label="Color" 
          name="color"
          options={jsonData.color_mapping}  
          onChange={handleChange}
        />        
        <Dropdown
          label="Color Intensity"         
          name="colorIntensity"         
          options={jsonData.color_intensity_mapping}         
          onChange={handleChange}
        />
        <Dropdown
          label="Clarity"
          name="clarity"
          options={jsonData.clarity_mapping}         
          onChange={handleChange}        
        />        
        <Dropdown
          label="Treatment"         
          name="treatment"         
          options={jsonData.treatment_mapping}         
          onChange={handleChange}         
        />        
        <Dropdown         
          label="Cut"         
          name="cut"         
          options={jsonData.cut_mapping}         
          onChange={handleChange}         
        />        
        <Dropdown         
          label="Type"         
          name="type"         
          options={jsonData.type_mapping}         
          onChange={handleChange}         
        />  
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 mt-10">
            <button type="button" onClick={handleClear} className="bg-transparent text-blue-500 font-semibold hover:text-blue-700 py-2 px-4 border border-blue-500 hover:border-blue-700 rounded">Clear</button>
            <button type="submit" className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Submit</button>
          </div>
          {error && <p className="text-red-500 mt-4">{error}</p>}
          {prediction !== null && <p className="mt-4 text-lg font-bold">Estimated Price: ${prediction}</p>}
        
        </div>
      </form>
    </div>
  );
};

export default PriceCalculator;
