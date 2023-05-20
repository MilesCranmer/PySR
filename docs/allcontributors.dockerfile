# Build with:
# docker build -t allcontributors -f docs/allcontributors.dockerfile .
# Run with:
# docker run --rm -v $(pwd):/workspace allcontributors generate
FROM node:bullseye-slim

RUN yarn add --dev all-contributors-cli

ENV HOME=/workspace
WORKDIR /workspace

ENTRYPOINT ["yarn", "all-contributors"]
CMD ["generate", "--config=/workspace/.all-contributorsrc"]
