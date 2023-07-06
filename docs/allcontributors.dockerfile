# Build with:
# docker build -t allcontributors -f docs/allcontributors.dockerfile .
# Run with:
# docker run --rm -v $(pwd):/workspace allcontributors generate
FROM node:20.2.0-bullseye-slim

RUN yarn add --dev all-contributors-cli

WORKDIR /workspace

ENTRYPOINT ["yarn", "all-contributors"]
CMD ["generate", "--config=/workspace/.all-contributorsrc"]
