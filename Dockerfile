# Use official lightweight Node.js image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package configurations
COPY package*.json ./

# Install production dependencies
RUN npm install --production

# Copy application source code
COPY . .

# Generate SSL certificates if cert.pem/key.pem don't exist
RUN apk add --no-cache openssl && \
    openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes -subj '/CN=localhost'

# Expose HTTPS port (3000) and plain HTTP debug port (3001)
EXPOSE 3000
EXPOSE 3001

# Start the application
CMD ["npm", "start"]
