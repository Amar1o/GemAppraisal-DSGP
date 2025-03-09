import { useState, useEffect } from "react";
import axios from "axios";
import FileUpload from "./FileUpload.jsx";
import Footer from "./Footer.jsx";
import Navbar from "./Navbar.jsx";
import Disclaimer from "./Disclaimer.jsx";
import Banner from "./Banner.jsx";

// Dropdown component
function Dropdown({ label, name, options, value, onChange }) {
  return (
    <div className="form-control w-full">
      <label className="form-control w-full max-w-xs block font-medium text-gray-700">{label}: </label>
      <div className="mt-2">
      <select 
        name={name} 
        value={value}
        onChange={onChange} 
        className="select shadow rounded py-2 px-3 text-gray-700 focus:outline-none focus:shadow-outline w-full"
      >
        <option value=""></option>
        {Object.entries(options).map(([key, value]) => (
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
      <label className="form-control w-full max-w-xs block font-medium text-gray-700">{label}: </label>
      <div className="mt-2">
      <input 
        name={name} 
        type='text' 
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
  const [resetFile, setResetFile] = useState(false); // Track if file needs to be reset
  const [formError, setFormError] = useState("");

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


  // Define allowed color options for each type
  const typeColorMapping = {
    "Pink Sapphire": ["Pink", "Pinkish Purple", "Purplish Pink", "Reddish Pink", "Pinkish Brown"],
    "Blue Sapphire": ["Blue", "Bluish Green", "Bluish Grey", "Bluish Purple", "Greenish Blue", "Greyish Blue", "Purplish Blue", "Color Change"],
    "Green Sapphire": ["Green", "Bluish Green", "Greenish Blue", "Greenish Yellow", "Yellowish Green"],
    "Padparadscha Sapphire": ["Padparadscha (Pinkish-Orange / Orangish-Pink)"],
    "Ruby": ["Red", "Orangish Red", "Purplish Red", "Pinkish Red"],
    "White Sapphire": ["White"],
    "Yellow Sapphire": ["Yellow", "Greenish Yellow", "Yellowish Brown", "Yellowish Orange", "Orangish Yellow", "Yellowish Green"],
    "Purple Sapphire": ["Purple", "Pinkish Purple", "Purplish Pink", "Purplish Red", "Bluish Purple", "Violet"],
  };

   // Filter color options based on selected type
   const getFilteredColorOptions = () => {
    const selectedType = selectedValues.type;
    if (selectedType && typeColorMapping[selectedType]) {
      // Filter the full color options to only include allowed colors
      const allowedColors = typeColorMapping[selectedType];
      return Object.fromEntries(
        Object.entries(jsonData.color_mapping).filter(([key]) =>
          allowedColors.includes(key)
        )
      );
    }
    // If no type is selected or no mapping exists, return all colors
    return jsonData.color_mapping;
  };
  

  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setSelectedValues((prev) => {
      const newValues = { ...prev, [name]: value };
      // Reset color if type changes
      if (name === "type") {
        newValues.color = ""; // Reset color when type changes
      }
      return newValues;
    });
  };

  // Validation for form
  const validateForm = () => {
    // Check if all required fields are filled
    for (const key in selectedValues) {
      if (!selectedValues[key]) {
        return false; // Field is missing
      }
    }
    // Ensure carat value is greater than 0.1
    if (selectedValues.carat < 0.1) {
      return false;
    }
    return true; // All fields are filled
  };


  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    if (!validateForm()) {
      setError("Please fill in all required fields and ensure carat value is greater than 0.");
      setLoading(false);
      return; // Stop further submission if validation fails
    }
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
  
 // Handle form clear
  const handleClear = (e) => {
    e.preventDefault();
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
    setFormError("");
    setError("");
    setPrediction(null);
  };

  // Function to format price
  const formatPrice = (price) => {
    if (typeof price !== "number") {
      return "N/A"; // Handle invalid prices
    }
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(price);
  };

  if (!jsonData) {
    return <p>Loading...</p>;
  }
  
  return (
    <>
    <Navbar />
    <Banner />
    <div className="md:pr-[20px] bg-white rounded-xl bg-opacity-60 backdrop-filter backdrop-blur-lg w-full max-w-6xl mx-auto flex flex-col mt-10  border border-gray-200 shadow-sm ">
      
      <form onSubmit={handleSubmit} className="w-full max-w-5xl mx-auto flex flex-col md:flex-row gap-10 mt-12 items-center md:items-start px-4 ">
      <div className="w-3/4">
        <FileUpload file={file} setFile={setFile} />
      </div>
        <div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-y-5 gap-x-10 w-full min-w-[390px] md:min-w-[500px]">
        <Dropdown className="w-full min-w-[500px]"        
          label="Type"         
          name="type"         
          options={jsonData.type_mapping}      
          value={selectedValues.type}   
          onChange={handleChange}         
        />  
        <InputText
        label="Carat" 
        name="carat"  
        value={selectedValues.carat}
        onChange={handleChange} 
      />
      
        <Dropdown
          label="Color" 
          name="color"
          options={getFilteredColorOptions()} // Dynamically filtered options
          value={selectedValues.color}
          onChange={handleChange}
          disabled={!selectedValues.type} // Disable if no type is selected
        />        
        <Dropdown
          label="Color Intensity"         
          name="colorIntensity"         
          options={jsonData.color_intensity_mapping}   
          value={selectedValues.colorIntensity}      
          onChange={handleChange}
        />
        <Dropdown
          label="Shape"
          name="shape"
          options={jsonData.shape_mapping}
          value={selectedValues.shape}
          onChange={handleChange}
        />
        <Dropdown
          label="Clarity"
          name="clarity"
          options={jsonData.clarity_mapping}         
          value={selectedValues.clarity}
          onChange={handleChange}        
        />     
        <Dropdown         
          label="Cut"         
          name="cut"         
          options={jsonData.cut_mapping}         
          value={selectedValues.cut}
          onChange={handleChange}         
        />   
      <Dropdown 
        label="Cut Quality" 
        name="cutQuality" 
        options={jsonData.cut_quality_mapping} 
        value={selectedValues.cutQuality}
        onChange={handleChange} 
      />
        
        <Dropdown
          label="Origin"
          name="origin"
          options={jsonData.origin_mapping}
          value={selectedValues.origin}
          onChange={handleChange}
        />
           
        <Dropdown
          label="Treatment"         
          name="treatment"         
          options={jsonData.treatment_mapping}         
          value={selectedValues.treatment}
          onChange={handleChange}         
        />        
             
        
        </div>
        <div className="flex flex-col md:grid md:grid-cols-2 md:gap-x-10 mt-10 w-full space-y-4 md:space-y-0">
            <button type="button" onClick={handleClear} className="bg-transparent text-gray-700 font-semibold hover:text-gray-500 py-2 px-4 border border-gray-700 hover:border-gray-500 rounded">
              Clear
            </button>
            <button type="submit" className="bg-gray-700 hover:bg-gray-500 text-white font-semibold py-2 px-4 rounded">Submit</button>
        </div>
        <div className="visibility-hidden min-h-[32px]">
        <p className="text-red-500 mt-4 visibility-hidden">{error}</p>
        <p className="text-red-500 mt-4 visibility-hidden">{formError}</p>
        <p className="text-right text-xl font-bold visibility-hidden">
          {prediction !== null ? `Estimated Price: ${formatPrice(prediction)}` : ""}
        </p>
        </div>
        </div>
      </form>
    </div>
    <Disclaimer />
    <Footer />
    </>
  );
};

export default PriceCalculator;
