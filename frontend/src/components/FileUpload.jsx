import { useState } from "react";

export default function FileUpload({ file, setFile }) {
  const handleFileChange = (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile) {
      setFile({
        file: uploadedFile,
        preview: URL.createObjectURL(uploadedFile),
      });
    }
  };

  return (
    <div className="flex flex-col h-full w-sm md:w-full mx-auto ">
      <label
        htmlFor="dropzone-file"
        className="flex flex-col items-center justify-center  h-64 md:h-[445px] lg:w-[475px]  border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 overflow-hidden"
      >
        {file ? (
          file.file.type === "video/mp4" ? (
            <video src={file.preview} controls className="w-full h-full object-cover" />
          ) : (
            <img src={file.preview} alt="Uploaded preview" className="w-full h-full object-cover" />
          )
        ) : (
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
            <svg
              className="w-8 h-8 mb-4 text-gray-500 dark:text-gray-400"
              aria-hidden="true"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 20 16"
            >
              <path
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
              />
            </svg>
            <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
              <span className="font-semibold">Click to upload</span> or drag and drop
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              PNG, JPG or MP4 (MAX. 800x400px)
            </p>
          </div>
        )}
        <input
          id="dropzone-file"
          type="file"
          accept=".jpg, .jpeg, .png, .mp4"
          className="hidden"
          onChange={handleFileChange}
        />
      </label>
    </div>
  );
}
