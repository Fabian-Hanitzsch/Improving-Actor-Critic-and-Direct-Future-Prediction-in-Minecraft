docker-compose build && docker tag minecraft-rl-basics-nodejs-mineflayer stfahani/minecraft-rl:nodejs-mineflayer && docker tag minecraft-rl-basics-environment-api stfahani/minecraft-rl:environment-api && docker tag minecraft-rl-basics-trainer stfahani/minecraft-rl:trainer && docker tag minecraft-rl-basics-server stfahani/minecraft-rl:server && docker push stfahani/minecraft-rl:trainer && docker push stfahani/minecraft-rl:environment-api && docker push stfahani/minecraft-rl:nodejs-mineflayer && docker push stfahani/minecraft-rl:server