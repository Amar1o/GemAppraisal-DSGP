function Banner() {
    return (
        <div className="relative font-sans before:absolute before:w-full before:h-full before:inset-0 before:bg-black before:opacity-50 before:z-10">
      <img src="/banner.jpg" alt="Banner Image" className="absolute inset-0 w-full h-full object-cover" />

      <div className="min-h-[350px] relative z-40 h-full max-w-6xl mx-auto flex flex-col justify-center items-center text-center text-white p-6 font-[Open Sans]">
        <h1 className="sm:text-5xl text-2xl font-bold mb-6">Price Calculator</h1>
        <p className="sm:text-xl text-xl text-center text-gray-200 ">Find an approximate market value for your corundum!</p>
      </div>
    </div>
    );
}   

export default Banner