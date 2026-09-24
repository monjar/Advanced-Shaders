import type { DemoEntry } from './demo';
import { cloudsEntry } from '../demos/clouds';
import { oceanEntry } from '../demos/ocean';
import { portalsEntry } from '../demos/portals';
import { sdfEntry } from '../demos/sdf';
import { watercolourEntry } from '../demos/watercolour';

/** Every shader study in the repo. Add new entries here. */
export const demos: DemoEntry[] = [oceanEntry, cloudsEntry, watercolourEntry, portalsEntry, sdfEntry];
