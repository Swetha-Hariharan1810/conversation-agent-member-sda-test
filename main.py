from contextlib import asynccontextmanager

from agent.app_graph import warm_llm_connections


@asynccontextmanager
async def lifespan(app):
    await warm_llm_connections()  # Blocking await — server accepts no traffic until warm-up completes
    yield


def main():
    print("Hello from conversation-agent-cigna-member-sdo-langgraph!")


if __name__ == "__main__":
    main()
