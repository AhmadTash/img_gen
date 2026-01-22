#!/bin/bash
# Netlify build script - ensures we're in the right directory
cd "$(dirname "$0")"
npm install
npm run build
