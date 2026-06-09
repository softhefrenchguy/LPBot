FROM node:20-alpine AS build
WORKDIR /app/dashboard
COPY dashboard/package.json ./package.json
RUN npm install
COPY dashboard/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dashboard/dist /usr/share/nginx/html
COPY dashboard/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 3000
