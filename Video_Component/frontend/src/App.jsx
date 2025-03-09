import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Login from './components/Login';
import SignUp from './components/SignUp';
import PriceCalculator from './components/PriceCalculator';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/price-calculator" element={<PriceCalculator />} />
      </Routes>
    </Router>
  );
};

export default App;
