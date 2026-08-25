#!/usr/bin/env python3
"""Small dependency-free TCP relay for localhost-only infrastructure bridges."""

from __future__ import annotations

import argparse
import asyncio
import logging


async def copy_stream(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    while True:
        data = await reader.read(64 * 1024)
        if not data:
            break
        writer.write(data)
        await writer.drain()


async def relay_connection(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    *,
    target_host: str,
    target_port: int,
) -> None:
    peer = client_writer.get_extra_info("peername")
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(
            target_host, target_port
        )
    except Exception:
        logging.exception("upstream connect failed peer=%s", peer)
        client_writer.close()
        await client_writer.wait_closed()
        return

    tasks = {
        asyncio.create_task(copy_stream(client_reader, upstream_writer)),
        asyncio.create_task(copy_stream(upstream_reader, client_writer)),
    }
    try:
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        for writer in (upstream_writer, client_writer):
            writer.close()
        await asyncio.gather(
            upstream_writer.wait_closed(),
            client_writer.wait_closed(),
            return_exceptions=True,
        )


async def serve(args: argparse.Namespace) -> None:
    server = await asyncio.start_server(
        lambda reader, writer: relay_connection(
            reader,
            writer,
            target_host=args.target_host,
            target_port=args.target_port,
        ),
        args.listen_host,
        args.listen_port,
    )
    addresses = ",".join(str(sock.getsockname()) for sock in server.sockets or ())
    logging.info(
        "relay listening=%s target=%s:%s", addresses, args.target_host, args.target_port
    )
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
