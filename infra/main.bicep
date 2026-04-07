@description('Container image to deploy')
param containerImage string = 'evcc/optimizer:latest'

@description('Azure region for all resources')
param location string = 'germanywestcentral'

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-optimizer-prod'
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
  }
}

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2025-02-01' = {
  name: 'optimizer-logs'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource containerAppEnv 'Microsoft.App/managedEnvironments@2025-01-01' = {
  name: 'optimizer-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource containerApp 'Microsoft.App/containerApps@2025-01-01' = {
  name: 'optimizer'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 7050
      }
      secrets: [
        {
          name: 'jwt-token-secret'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/jwt-token-secret'
          identity: 'system'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'optimizer'
          image: containerImage
          resources: {
            cpu: json('2')
            memory: '4Gi'
          }
          env: [
            { name: 'OPTIMIZER_TIME_LIMIT', value: '25' }
            { name: 'OPTIMIZER_NUM_THREADS', value: '1' }
            { name: 'GUNICORN_CMD_ARGS', value: '--workers 4 --timeout 60 --max-requests 32 --max-requests-jitter 8' }
            { name: 'JWT_TOKEN_SECRET', secretRef: 'jwt-token-secret' }
          ]
          probes: [
            {
              type: 'startup'
              tcpSocket: {
                port: 7050
              }
              periodSeconds: 5
              failureThreshold: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 5
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
}


output fqdn string = containerApp.properties.configuration.ingress.fqdn
