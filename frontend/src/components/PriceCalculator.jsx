import { useState, useEffect } from "react";

function Dropdown({ label, name, options, onChange }) {
  return (
    <div className="form-control w-full max-w-xs">
      <label className="form-control w-full max-w-xs block font-medium text-gray-700">{label}: </label>
      <div className="mt-2">
      <select name={name} onChange={onChange} className="select shadow rounded w-full py-2 px-3 text-gray-700 focus:outline-none focus:shadow-outline">
        <option value="" disabled selected></option>
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
  const [selectedValues, setSelectedValues] = useState({
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
    fetch("/encoded_features.json")
      .then((response) => response.json())
      .then((data) => setJsonData(data))
      .catch((error) => console.error("Error fetching JSON:", error));
  }, []);

  const handleChange = (e) => {
    setSelectedValues({ ...selectedValues, [e.target.name]: e.target.value });
  };

  if (!jsonData) {
    return <p>Loading...</p>;
  }


  return (
    <div>
      <form className="w-full max-w-lg">
        <div>
          <div class="flex items-center justify-center w-full">
              <label for="dropzone-file" class="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-gray-800 dark:bg-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-600">
                  <div class="flex flex-col items-center justify-center pt-5 pb-6">
                      <svg class="w-8 h-8 mb-4 text-gray-500 dark:text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                          <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
                      </svg>
                      <p class="mb-2 text-sm text-gray-500 dark:text-gray-400"><span class="font-semibold">Click to upload</span> or drag and drop</p>
                      <p class="text-xs text-gray-500 dark:text-gray-400">SVG, PNG, JPG or GIF (MAX. 800x400px)</p>
                  </div>
                  <input id="dropzone-file" type="file" class="hidden" />
              </label>
          </div> 
        </div>
        <div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
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
        <button type="submit" className="bg-transparent  text-blue-500 font-semibold hover:text-blue-700 py-2 px-4 border border-blue-500 hover:border-blue-700 rounded">Clear</button>  
        <button type="submit" className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Submit</button>  
        </div>
        </div>
      </form>
    </div>
  );
};

export default PriceCalculator;
