#!/usr/bin/env python3
"""
Example: Create and execute a debate workflow using AgentMesh.

This demonstrates the full AgentMesh stack:
1. Create a workflow with Hidden Layer debate strategy
2. Execute it via the API
3. View the results

Prerequisites:
- Docker Compose running (postgres, redis)
- AgentMesh API server running
"""

import asyncio
import json
from uuid import uuid4

import httpx


API_BASE = "http://localhost:8000/api"


async def main():
    """Run example workflow"""

    async with httpx.AsyncClient() as client:
        print("🚀 AgentMesh Example: Debate Workflow\n")

        # Step 1: Create organization (normally done via auth)
        org_id = str(uuid4())
        print(f"📦 Using organization: {org_id}\n")

        # Step 2: Create workflow
        print("📝 Creating debate workflow...")

        workflow_request = {
            "name": "Renewable Energy Debate",
            "description": "Multi-agent debate on renewable energy investment",
            "org_id": org_id,
            "graph": {
                "nodes": [
                    {
                        "id": "start",
                        "type": "start",
                        "label": "Start",
                        "config": {}
                    },
                    {
                        "id": "debate1",
                        "type": "strategy",
                        "label": "Debate (3 agents)",
                        "strategy_id": "debate",
                        "config": {
                            "n_debaters": 3,
                            "n_rounds": 2
                        }
                    },
                    {
                        "id": "end",
                        "type": "end",
                        "label": "End",
                        "config": {}
                    }
                ],
                "edges": [
                    {
                        "id": "e1",
                        "from_node_id": "start",
                        "to_node_id": "debate1"
                    },
                    {
                        "id": "e2",
                        "from_node_id": "debate1",
                        "to_node_id": "end"
                    }
                ]
            }
        }

        response = await client.post(
            f"{API_BASE}/workflows",
            json=workflow_request
        )
        response.raise_for_status()
        workflow = response.json()

        workflow_id = workflow["id"]
        print(f"✅ Workflow created: {workflow_id}")
        print(f"   Name: {workflow['name']}")
        print(f"   Nodes: {len(workflow['graph']['nodes'])}\n")

        # Step 3: Execute workflow
        print("🎯 Executing workflow...")
        print("   Task: Should we invest in renewable energy?")
        print("   Strategy: Debate (Hidden Layer research!)")
        print("   Provider: Anthropic Claude\n")

        run_request = {
            "input": {
                "task": "Should we invest in renewable energy? Consider economic, environmental, and technological factors."
            },
            "context": {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
                "temperature": 0.7
            }
        }

        response = await client.post(
            f"{API_BASE}/workflows/{workflow_id}/runs",
            json=run_request
        )
        response.raise_for_status()
        run = response.json()

        run_id = run["id"]
        print(f"✅ Run completed: {run_id}")
        print(f"   Status: {run['status']}")

        if run['status'] == 'succeeded':
            print(f"\n📊 Results:")
            print(f"   Output: {run['output'][:200]}...")

            # Step 4: Get detailed steps
            response = await client.get(f"{API_BASE}/runs/{run_id}/steps")
            response.raise_for_status()
            steps = response.json()

            print(f"\n🔍 Execution Steps:")
            for step in steps:
                print(f"   - {step['node_id']} ({step['node_type']}): {step['status']}")
                if step['latency_s']:
                    print(f"     Latency: {step['latency_s']:.2f}s")
                if step['tokens_in']:
                    print(f"     Tokens: {step['tokens_in']} in, {step['tokens_out']} out")
                if step['cost_usd']:
                    print(f"     Cost: ${step['cost_usd']:.4f}")

        else:
            print(f"\n❌ Run failed:")
            print(f"   Error: {run.get('error', 'Unknown error')}")

        print(f"\n✨ Example complete!")
        print(f"\nView in API docs: http://localhost:8000/docs")
        print(f"Get run: curl {API_BASE}/runs/{run_id}")


if __name__ == "__main__":
    asyncio.run(main())
