import React from "react";
import Navbar from "./Navbar";
import Footer from "./Footer";
import { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";

// Hero Section
const Hero = () => {
  // Define an array of background colors
  const colors = ["/front.webp", "/banner.jpg", "/signup_img.webp"]; // Tailwind Hex Colors
  const [currentIndex, setCurrentIndex] = useState(0);

  // Auto-change background color every 3 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % colors.length);
    }, 3000); // Change color every 3 seconds

    return () => clearInterval(interval); // Cleanup on unmount
  }, [currentIndex]);

  return (
    <section 
      className="relative h-[500px] flex items-center justify-center text-white text-center transition-all duration-1000 before:absolute before:w-full before:h-full before:inset-0 before:bg-black before:opacity-50 before:z-5"
      style={{ backgroundImage: `url(${colors[currentIndex]})`, backgroundSize: "cover" }} // ✅ Background color changes dynamically
    >
      {/* Dark Overlay for Readability */}
      <div className="absolute inset-0 bg-blue bg-opacity-30 pointer-events-none"></div>

      {/* Hero Content */}
      <div className="z-10 relative font-[Inter]">
        <h1 className="text-6xl">Welcome to Our Website</h1>
        <p className="mt-4 text-xl">Discover more about us, our team, and how to contact us.</p>
        <a href="#about" className="mt-6 inline-block bg-white text-blue-600 hover:bg-gray-300 px-6 py-2 rounded-full text-xl font-semibold transition-colors duration-300">
          Learn More
        </a>
      </div>
    </section>
  );
};




// About Section
const About = () => (
  <section id="about" className="py-16 px-6 text-center bg-gray-50">
    <h2 className="text-3xl font-bold text-gray-800">About Us</h2>
    <div className="mt-4 max-w-3xl mx-auto text-gray-600 leading-relaxed text-lg">
      
      <p className="mb-4">
        We are second-year undergraduate students pursuing a degree in 
        Artificial Intelligence and Data Science. As part of our 
        second-year project, we have designed an advanced 
        Gemstone Appraisal System, specifically tailored for 
        <strong> sapphires and rubies</strong>.
      </p>

      <p className="mb-4">
        Our system leverages  
        <strong> data science techniques</strong> to analyze gemstones and provide 
        accurate market valuations. By simply uploading an 
        <strong> image or video</strong> of a sapphire or ruby, our AI model evaluates 
        its characteristics and determines its 
        <strong> current market value</strong>.
      </p>

      <p>
        We aim to revolutionize the gemstone industry by making 
        appraisals faster, more accessible, and data-driven. Our project bridges 
        the gap between technology and gemology, ensuring that 
        buyers, sellers, and enthusiasts can make 
        informed decisions with confidence.
      </p>

    </div>
  </section>
);




// Team Section
const Team = () => {
  const teamMembers = [
    { name: "Khadheeja", role: "Founder" },
    { name: "Amar", role: "Founder" },
    { name: "Sarasi", role: "Founder" },
    { name: "Norman", role: "Founder" },
  ];

  // Array of colors for the logos
  const colors = ["bg-blue-500", "bg-red-500", "bg-green-500", "bg-purple-500"];

  return (
    <section id="team" className="py-16 px-6 text-center">
      <h2 className="text-3xl font-bold text-gray-800">Meet Our Team</h2>
      <div className="mt-8 flex flex-wrap justify-center gap-8">
        {teamMembers.map((member, index) => (
          <div key={index} className="bg-white p-4 rounded-lg shadow-md w-64 flex flex-col items-center">
            {/* Logo with First Letter (Different Colors) */}
            <div className={`w-20 h-20 rounded-full text-white flex items-center justify-center text-3xl font-bold ${colors[index % colors.length]}`}>
              {member.name.charAt(0)}
            </div>

            <h3 className="mt-4 text-lg font-semibold">{member.name}</h3>
            <p className="text-gray-600">{member.role}</p>
          </div>
        ))}
      </div>
    </section>
  );
};

// Contact Section
const Contact = () => (
  <section id="contact" className="py-16 px-6 text-center bg-gray-50">
    <h2 className="text-3xl font-bold text-gray-800">Contact Us</h2>
    <p className="mt-4 text-gray-600">We’d love to hear from you! Fill out the form below.</p>
    <form className="mt-6 max-w-lg mx-auto bg-white p-6 rounded-lg shadow-md">
      <div className="mb-4">
        <input type="text" placeholder="Your Name" className="w-full p-3 border border-gray-300 rounded"/>
      </div>
      <div className="mb-4">
        <input type="email" placeholder="Your Email" className="w-full p-3 border border-gray-300 rounded"/>
      </div>
      <div className="mb-4">
        <textarea placeholder="Your Message" className="w-full p-3 border border-gray-300 rounded"></textarea>
      </div>
      <button type="submit" className="w-full bg-cyan-500 hover:bg-cyan-700 text-white py-2 rounded">Send Message</button>
    </form>
  </section>
);

// Home Page (Keep only one default export)
const Home = () => {
  const location = useLocation();

  useEffect(() => {
    if (location.state?.scrollTo) {
      const section = document.getElementById(location.state.scrollTo);
      if (section) {
        setTimeout(() => {
          section.scrollIntoView({ behavior: "smooth" });
        }, 100); // Small delay for a smoother effect
      }
    }
  }, [location]);

  return (
    <div>
      <Navbar />
      <Hero />
      <About id="about" />
      <Team id="team" />
      <Contact id="contact" />
      <Footer />
    </div>
  );
};
export { Hero, About, Team, Contact }; // ✅ Named Exports
export default Home; // ✅ Default Export for Home
