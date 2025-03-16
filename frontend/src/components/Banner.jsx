function Banner() {
    return (
        <div className="top-0 relative font-[Inter] before:absolute before:w-full before:h-full before:inset-0 before:bg-black before:opacity-50 before:z-10">
      <img src="/front.webp" alt="Banner Image" className="absolute inset-0 w-full h-full object-cover  " />

      <div className="min-h-[500px] h-[500px] top-0 relative z-30  max-w-6xl mx-auto flex flex-col justify-center items-center text-center text-white p-6 font-[Inter]">
        <h1 className="sm:text-6xl text-4xl mb-6">Price Calculator</h1>
        <p className="sm:text-xl text-xl text-center text-gray-200 ">Find an approximate market value for your corundum!</p>
      </div>
    </div>
    );
}   

export default Banner