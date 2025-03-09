import React, { useState } from 'react';
import { supabase } from '../supabaseClient.js';
import Footer from "./Footer.jsx";

const SignUp = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleSignUp = async (e) => {
    e.preventDefault();
    const { error } = await supabase.auth.signUp({ email, password });
    if (error) {
      setError(error.message);
    } else {
      setSuccess('Sign-up successful! Check your email for confirmation.');
      setError('');
    }
  };

  return (
    <>
      <div className="flex justify-center py-4">
        <img
          alt="Your Company"
          src="/logo.png"
          className="h-20 w-auto"
        />
      </div>
      <div className="flex min-h-screen flex-1 flex-col justify-center px-6 py-12 lg:px-8 bg-gradient-to-r from-sky-200 via-blue-300 to-sky-400">
        <div className="sm:mx-auto sm:w-full sm:max-w-md bg-white p-8 rounded-lg shadow-lg border border-gray-300">
          <h2 className="mt-6 text-center text-2xl font-bold tracking-tight text-gray-900">
            Sign up to create an account
          </h2>
          {error && <div className="mt-4 text-red-500 text-sm text-center">{error}</div>}
          {success && <div className="mt-4 text-green-500 text-sm text-center">{success}</div>}
          <form onSubmit={handleSignUp} className="mt-6 space-y-6">
            <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-900">
                  Email address
                </label>
                <div className="mt-1">
                  <input
                    type="email"
                    placeholder="Email"
                    className="block w-full rounded-md bg-white px-3 py-2 text-gray-900 shadow-sm border border-gray-300 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>
            </div>
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-900">
                Password
              </label>
              <div className="mt-1">
                <input
                  type="password"
                  placeholder="Password"
                  className="block w-full rounded-md bg-white px-3 py-2 text-gray-900 shadow-sm border border-gray-300 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
            </div>
            <div>
              <button
                type="submit"
                className="w-full flex justify-center rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
              >
                Sign up
              </button>
            </div>
          </form>
          <p className="mt-6 text-center text-sm text-gray-500">
            Already a member?{' '}
            <a href="/" className="font-semibold text-indigo-600 hover:text-indigo-500">
              Sign in to your account
            </a>
          </p>
        </div>
      </div>
      <Footer />
    </>
  );
};

export default SignUp;
