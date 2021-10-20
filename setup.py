from setuptools import setup

setup(
    name="grant",
    version="1.0",
    description="Grant Hussey personal tools",
    url="resist.bot",
    author="Grant Hussey",
    author_email="grant.hussey@nyulangone.org",
    license=" MIT ",
    packages=["grant", "colorpackage"],
    install_requires=["matplotlib", "pandas"],
    zip_safe=False,
)
